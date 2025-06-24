import os
import sys
import json
import gc
import signal
import psutil
import time
import atexit
from datetime import datetime

# Simple local imports since all files are in the same directory
from config import (
    get_metadata_dir, get_video_metadata_path, get_audio_metadata_path,
    get_text_metadata_path, get_image_metadata_path, get_memory_log_path,
    get_failed_files_path, migrate_existing_metadata, cleanup_temp_frames,
    print_directory_structure
)

def get_memory_usage():
    """Get current memory usage in MB"""
    process = psutil.Process()
    return process.memory_info().rss / 1024 / 1024

def load_memory_log(log_file=None):
    """Load existing memory log from JSON file"""
    if log_file is None:
        log_file = get_memory_log_path()
    try:
        with open(log_file, 'r') as f:
            return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        return {"session_start": datetime.now().isoformat(), "entries": []}

def log_memory_to_file(file_path, action_type, memory_before, memory_after, file_index, total_files, status="success", error_msg=None, log_file=None):
    """Log memory usage and action to JSON file"""
    if log_file is None:
        log_file = get_memory_log_path()
    
    memory_log = load_memory_log(log_file)
    
    memory_used = memory_after - memory_before
    
    entry = {
        "timestamp": datetime.now().isoformat(),
        "file_path": file_path,
        "file_name": os.path.basename(file_path),
        "action_type": action_type,  # "video", "text", "audio", "image"
        "memory_before_mb": round(memory_before, 1),
        "memory_after_mb": round(memory_after, 1),
        "memory_used_mb": round(memory_used, 1),
        "file_index": file_index + 1,
        "total_files": total_files,
        "status": status,  # "success" or "failed"
        "error_msg": error_msg
    }
    
    memory_log["entries"].append(entry)
    
    # Save to file
    with open(log_file, 'w') as f:
        json.dump(memory_log, f, indent=2)
    
    return entry

def log_memory_usage(file_path, memory_before, file_index, total_files, action_type="unknown", status="success", error_msg=None, log_file=None):
    """Log detailed memory usage after processing a file"""
    if log_file is None:
        log_file = get_memory_log_path()
        
    memory_after = get_memory_usage()
    memory_used = memory_after - memory_before
    
    # Log to JSON file
    entry = log_memory_to_file(file_path, action_type, memory_before, memory_after, file_index, total_files, status, error_msg, log_file)
    
    # Print to console
    status_emoji = "✅" if status == "success" else "❌"
    print(f"💾 Memory for {os.path.basename(file_path)} ({action_type}): {memory_before:.1f} → {memory_after:.1f} MB ({memory_used:+.1f} MB) ({file_index + 1}/{total_files}) {status_emoji}")
    
    # Alert if memory usage is getting high
    if memory_after > 2000:  # 2GB
        print(f"⚠️  HIGH MEMORY USAGE: {memory_after:.1f} MB - Consider restarting batch processing")
    elif memory_after > 1000:  # 1GB
        print(f"🔶 Elevated memory usage: {memory_after:.1f} MB")
    
    # Alert for high memory consumption in single operation
    if memory_used > 100:  # 100MB+ for single file
        print(f"⚠️  HIGH MEMORY CONSUMPTION: {memory_used:.1f} MB for single file")
    elif memory_used > 50:  # 50MB+ for single file
        print(f"🔶 Elevated memory consumption: {memory_used:.1f} MB for single file")
    
    return memory_after

def get_memory_summary(log_file=None):
    """Get summary statistics from memory log"""
    if log_file is None:
        log_file = get_memory_log_path()
        
    memory_log = load_memory_log(log_file)
    entries = memory_log.get("entries", [])
    
    if not entries:
        return None
    
    # Handle both old and new format
    memory_after = []
    memory_used = []
    
    for entry in entries:
        if "memory_after_mb" in entry:
            # New format
            memory_after.append(entry["memory_after_mb"])
            memory_used.append(entry["memory_used_mb"])
        else:
            # Old format fallback
            memory_after.append(entry.get("memory_mb", 0))
            memory_used.append(0)
    
    action_counts = {}
    status_counts = {"success": 0, "failed": 0}
    
    for entry in entries:
        action_type = entry["action_type"]
        action_counts[action_type] = action_counts.get(action_type, 0) + 1
        status_counts[entry["status"]] += 1
    
    return {
        "session_start": memory_log["session_start"],
        "total_entries": len(entries),
        "memory": {
            "min_after": min(memory_after) if memory_after else 0,
            "max_after": max(memory_after) if memory_after else 0,
            "latest_after": memory_after[-1] if memory_after else 0,
            "total_growth": memory_after[-1] - memory_after[0] if len(memory_after) > 1 else 0,
            "total_used": sum(memory_used),
            "avg_used_per_file": sum(memory_used) / len(memory_used) if memory_used else 0,
            "max_used_single_file": max(memory_used) if memory_used else 0
        },
        "actions": action_counts,
        "status": status_counts
    }

# Process management constants
def get_pid_file_path():
    """Get the path for the PID file in the metadata directory"""
    return os.path.join(get_metadata_dir(), "tagdirectory.pid")

def get_lock_file_path():
    """Get the path for the lock file in the metadata directory"""
    return os.path.join(get_metadata_dir(), "tagdirectory.lock")

def get_current_pid():
    """Get the current process ID"""
    return os.getpid()

def write_pid_file():
    """Write current PID to file"""
    with open(get_pid_file_path(), 'w') as f:
        f.write(str(get_current_pid()))

def read_pid_file():
    """Read PID from file, return None if file doesn't exist"""
    try:
        with open(get_pid_file_path(), 'r') as f:
            return int(f.read().strip())
    except (FileNotFoundError, ValueError):
        return None

def is_process_running(pid):
    """Check if a process with given PID is running"""
    try:
        process = psutil.Process(pid)
        # Check if it's actually our tagdirectory process
        cmdline = ' '.join(process.cmdline())
        return 'tagdirectory.py' in cmdline
    except (psutil.NoSuchProcess, psutil.AccessDenied):
        return False

def kill_existing_process(pid):
    """Kill existing tagdirectory process"""
    try:
        process = psutil.Process(pid)
        cmdline = ' '.join(process.cmdline())
        
        if 'tagdirectory.py' in cmdline:
            print(f"🔄 Found existing tagdirectory process (PID: {pid})")
            print(f"   Command: {cmdline}")
            print(f"🛑 Terminating existing process...")
            
            # Try graceful termination first
            process.terminate()
            
            # Wait up to 10 seconds for graceful shutdown
            try:
                process.wait(timeout=10)
                print(f"✅ Existing process terminated gracefully")
            except psutil.TimeoutExpired:
                # Force kill if it doesn't terminate gracefully
                print(f"⚡ Force killing unresponsive process...")
                process.kill()
                process.wait(timeout=5)
                print(f"✅ Existing process force killed")
            
            return True
        else:
            print(f"⚠️  PID {pid} exists but is not tagdirectory.py process")
            return False
            
    except (psutil.NoSuchProcess, psutil.AccessDenied) as e:
        print(f"⚠️  Could not access process {pid}: {e}")
        return False

def cleanup_on_exit():
    """Clean up PID file on exit"""
    try:
        if os.path.exists(get_pid_file_path()):
            os.remove(get_pid_file_path())
        if os.path.exists(get_lock_file_path()):
            os.remove(get_lock_file_path())
    except:
        pass

def initialize_memory_log(log_file=None):
    """Initialize a new memory log session"""
    if log_file is None:
        log_file = get_memory_log_path()
    
    memory_log = {
        "session_start": datetime.now().isoformat(),
        "entries": []
    }
    
    with open(log_file, 'w') as f:
        json.dump(memory_log, f, indent=2)
    
    print(f"📝 Memory logging initialized: {log_file}")
    return memory_log

def setup_signal_handlers():
    """Set up signal handlers for graceful shutdown"""
    def signal_handler(signum, frame):
        print(f"\n🛑 Received signal {signum}, shutting down gracefully...")
        cleanup_on_exit()
        sys.exit(0)
    
    signal.signal(signal.SIGINT, signal_handler)   # Ctrl+C
    signal.signal(signal.SIGTERM, signal_handler)  # Termination request

def check_status():
    """Check if tagdirectory is currently running"""
    existing_pid = read_pid_file()
    
    if existing_pid:
        if is_process_running(existing_pid):
            try:
                process = psutil.Process(existing_pid)
                create_time = datetime.fromtimestamp(process.create_time())
                cmdline = ' '.join(process.cmdline())
                
                print(f"✅ tagdirectory.py is RUNNING")
                print(f"   PID: {existing_pid}")
                print(f"   Started: {create_time.strftime('%Y-%m-%d %H:%M:%S')}")
                print(f"   Command: {cmdline}")
                print(f"   Memory: {process.memory_info().rss / 1024 / 1024:.1f} MB")
                return True
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                print(f"❌ PID file exists but process {existing_pid} is not accessible")
                return False
        else:
            print(f"❌ PID file exists but process {existing_pid} is not running")
            return False
    else:
        print(f"✅ No tagdirectory.py process is currently running")
        return False

def stop_running_instance():
    """Stop any running tagdirectory instance"""
    existing_pid = read_pid_file()
    
    if existing_pid:
        if is_process_running(existing_pid):
            print(f"🛑 Stopping tagdirectory process (PID: {existing_pid})")
            if kill_existing_process(existing_pid):
                cleanup_on_exit()
                print(f"✅ Process stopped successfully")
                return True
            else:
                print(f"❌ Failed to stop process")
                return False
        else:
            print(f"⚠️  PID file exists but process is not running")
            cleanup_on_exit()
            return True
    else:
        print(f"✅ No tagdirectory process is running")
        return True

def ensure_single_instance():
    """Ensure only one instance of tagdirectory.py is running"""
    print("🔍 Checking for existing tagdirectory processes...")
    
    # Check PID file first
    existing_pid = read_pid_file()
    
    if existing_pid:
        if is_process_running(existing_pid):
            print(f"🔄 Existing process found (PID: {existing_pid})")
            if kill_existing_process(existing_pid):
                time.sleep(2)  # Give it time to fully shut down
            else:
                print(f"❌ Failed to terminate existing process")
        else:
            print(f"🧹 Cleaning up stale PID file (process {existing_pid} not running)")
    
    # Double-check: look for any tagdirectory processes by name
    current_pid = get_current_pid()
    killed_any = False
    
    for proc in psutil.process_iter(['pid', 'cmdline']):
        try:
            cmdline = ' '.join(proc.info['cmdline'] or [])
            if 'tagdirectory.py' in cmdline and proc.info['pid'] != current_pid:
                print(f"🔄 Found additional tagdirectory process (PID: {proc.info['pid']})")
                print(f"   Command: {cmdline}")
                proc.terminate()
                killed_any = True
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            continue
    
    if killed_any:
        print("⏳ Waiting for processes to terminate...")
        time.sleep(3)
    
    # Write our PID file
    write_pid_file()
    print(f"✅ This process is now the active tagdirectory instance (PID: {current_pid})")
    
    # Set up cleanup
    atexit.register(cleanup_on_exit)
    setup_signal_handlers()

def is_video_file(filename):
    video_extensions = [".mp4", ".mov", ".mkv", ".avi", ".webm"]
    return any(filename.lower().endswith(ext) for ext in video_extensions)

def is_text_file(filename):
    text_extensions = [".pdf", ".docx", ".xlsx", ".xls", ".txt"]
    return any(filename.lower().endswith(ext) for ext in text_extensions)

def is_audio_file(filename):
    audio_extensions = [".mp3", ".wav", ".m4a", ".flac", ".ogg"]
    return any(filename.lower().endswith(ext) for ext in audio_extensions)

def is_image_file(filename):
    image_extensions = [".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".webp"]
    return any(filename.lower().endswith(ext) for ext in image_extensions)

def load_failed_files(failed_files_json=None):
    """Load the list of previously failed files"""
    if failed_files_json is None:
        failed_files_json = get_failed_files_path()
        
    if os.path.exists(failed_files_json):
        with open(failed_files_json, 'r') as f:
                return json.load(f)
    return {"failed_files": []}

def save_failed_files(failed_files, failed_files_json=None):
    """Save the list of failed files"""
    if failed_files_json is None:
        failed_files_json = get_failed_files_path()
    
    with open(failed_files_json, 'w') as f:
        json.dump({"failed_files": failed_files}, f, indent=2)

def should_retry_failed_file(file_path, max_attempts=3, failed_file=None):
    """Check if a failed file should be retried"""
    if failed_file is None:
        failed_file = get_failed_files_path()
    failed_files = load_failed_files(failed_file)
    abs_path = os.path.abspath(file_path)
    
    if abs_path not in failed_files:
        return True  # Not failed before, can try
    
    attempts = failed_files[abs_path].get("attempts", 0)
    return attempts < max_attempts

def load_metadata(filename):
        """Load metadata from JSON file, return empty dict if file doesn't exist"""
        try:
            if os.path.exists(filename):
                with open(filename, 'r') as f:
                    return json.load(f)
        except Exception as e:
            print(f"⚠️ Error loading {filename}: {e}")
        return {}

def batch_process_files(directory, progress_callback=None, stop_flag=None):
    """
    Process all supported files in a directory with memory monitoring and resume capability
    
    Args:
        directory: Directory path to process
        progress_callback: Optional callback function for progress updates (stage, progress, message)
        stop_flag: Optional callable that returns True when processing should stop
    """
    def emit_progress(stage, progress, message):
        """Helper function to emit progress if callback is provided"""
        if progress_callback:
            progress_callback(stage, progress, message)
        # Always print to console as well
        if stage == "directory_processing":
            print(f"📁 {progress:.1f}% - {message}")
    
    print("🚀 Starting ContentCache batch processing...")
    emit_progress("directory_processing", 0, f"Starting directory processing: {directory}")
    
    # Initialize configuration and migrate existing metadata
    print("📁 Setting up cache directories...")
    print_directory_structure()
    emit_progress("directory_processing", 5, "Setting up cache directories...")
    
    # Migrate existing metadata files to cache directory
    migrated_files = migrate_existing_metadata()
    
    # Clean up any leftover temp frame directories
    cleanup_temp_frames()
    
    print("✅ Setup complete!\n")
    emit_progress("directory_processing", 10, "Setup complete, scanning for files...")
    
    start_memory = get_memory_usage()
    log_file = get_memory_log_path()
    failed_files_json = get_failed_files_path()
    
    print(f"💾 Starting memory usage: {start_memory:.1f} MB")
    print(f"📊 Memory log: {log_file}")
    print(f"❌ Failed files log: {failed_files_json}")
    
    # Initialize memory log for this session
    memory_log = load_memory_log(log_file)
    print(f"📈 Loading session from: {memory_log['session_start']}")
    
    # Load list of previously failed files to avoid re-processing
    failed_files = load_failed_files(failed_files_json)
    print(f"⚠️  Found {len(failed_files.get('failed_files', []))} previously failed files to skip")
    
    stats = {
        "processed": 0,
        "skipped": 0,
        "failed": 0,
        "videos": 0,
        "images": 0,
        "audio": 0,
        "text": 0,
        "start_time": time.time(),
        "errors": [],
        "memory_stats": {
            "initial": start_memory,
            "peak": start_memory,
            "current": start_memory
        }
    }

    # Import heavy modules only when actually processing
    print("📦 Loading processing modules...")
    from videotagger import tag_video_smart_conflict_resolution
    from imageprocessor import tag_image
    from audioanalyzer import analyze_audio_with_openai
    from textprocessor import TextProcessor
    
    text_processor = TextProcessor()
    
    # Helper functions for file processing (defined after processors are created)
    def determine_file_type(file):
        """Determine the type of file based on its extension"""
        if is_video_file(file):
            return "video"
        elif is_text_file(file):
            return "text"
        elif is_audio_file(file):
            return "audio"
        elif is_image_file(file):
            return "image"
        else:
            return "unknown"

    def process_file_by_type(file_type, file_path):
        """Process a file based on its determined type"""
        if file_type == "video":
            tag_video_smart_conflict_resolution(file_path, use_moondream_api=True, stop_flag=stop_flag)
            return "video"
        elif file_type == "text":
            text_processor.process_file(file_path)
            return "text"
        elif file_type == "audio":
            result = analyze_audio_with_openai(file_path)
            print(json.dumps(result, indent=2))
            return "audio"
        elif file_type == "image":
            tag_image(file_path)
            return "image"
        else:
            raise ValueError(f"Unsupported file type: {file_type}")
    
    # Load all metadata upfront - JSON files are small, so this is efficient
    video_metadata = load_metadata(get_video_metadata_path())
    audio_metadata = load_metadata(get_audio_metadata_path())
    text_metadata = load_metadata(get_text_metadata_path())
    image_metadata = load_metadata(get_image_metadata_path())
    
    # First pass: collect all files to process
    print("📝 Collecting files to process...")
    files_to_process = []
    
    for root, _, files in os.walk(directory):
        # Check for stop signal during file collection
        if stop_flag and callable(stop_flag) and stop_flag():
            print("⚠️ Processing stopped during file collection")
            emit_progress("directory_processing", 100, "Processing stopped by user")
            stop_running_instance()
            return {"status": "stopped", "message": "Processing stopped during file collection"}
            
        for file in files:
            file_path = os.path.join(root, file)
            abs_path = os.path.abspath(file_path)
            
            # Skip if not a supported file type
            if not (is_video_file(file) or is_audio_file(file) or is_text_file(file) or is_image_file(file)):
                continue
                
            files_to_process.append((file_path, abs_path, file))
    
    total_files = len(files_to_process)
    print(f"📁 Found {total_files} files to process")
    emit_progress("directory_processing", 15, f"Found {total_files} files to process")
    
    if total_files == 0:
        emit_progress("directory_processing", 100, "No supported files found in directory")
        print("✅ No supported files found to process")
        return
    
    # Initialize memory logging
    initialize_memory_log()
    
    # Get initial memory usage
    initial_memory = get_memory_usage()
    print(f"💾 Initial memory usage: {initial_memory:.1f} MB")
    
    # Second pass: process files with enhanced memory monitoring
    for file_index, (file_path, abs_path, file) in enumerate(files_to_process):
            
            # Check for stop signal at the beginning of each file processing
            if stop_flag and callable(stop_flag) and stop_flag():
                print(f"⚠️ Processing stopped by user after {stats['processed']} files")
                emit_progress("directory_processing", 100, f"Processing stopped by user. Processed {stats['processed']} files.")
                stop_running_instance()
                return {
                    "status": "stopped",
                    "total_files": total_files,
                    "processed": stats["processed"],
                    "skipped": stats["skipped"],
                    "failed": stats["failed"],
                    "memory_stats": stats["memory_stats"],
                    "errors": stats["errors"]
                }
            
            # Calculate progress percentage (15% to 95%, leaving 5% for final cleanup)
            file_progress = 15 + ((file_index / total_files) * 80)
            emit_progress("directory_processing", file_progress, f"Processing {file_index + 1}/{total_files}: {os.path.basename(file_path)}")
            
            # Check if file is already processed
            if is_video_file(file):
                if abs_path in video_metadata:
                    print(f"{file_path} is already tagged (video)")
                    stats["skipped"] += 1
                    continue
            elif is_text_file(file):
                if abs_path in text_metadata:
                    print(f"{file_path} is already tagged (text)")
                    stats["skipped"] += 1
                    continue
            elif is_audio_file(file):
                if abs_path in audio_metadata:
                    print(f"{file_path} is already tagged (audio)")
                    stats["skipped"] += 1
                    continue
            elif is_image_file(file):
                if abs_path in image_metadata:
                    print(f"{file_path} is already tagged (image)")
                    stats["skipped"] += 1
                    continue
            
            # Check if file has failed too many times
            if not should_retry_failed_file(abs_path):
                failed_info = failed_files.get(abs_path, {})
                attempts = failed_info.get("attempts", 0)
                last_error = failed_info.get("error", "Unknown error")
                print(f"⏭️  Skipping {file_path} - failed {attempts} times. Last error: {last_error}")
                stats["skipped"] += 1
                continue
                    
            print(f"\n🔍 Processing: {file_path} ({file_index + 1}/{total_files})")
            
            # Capture memory before processing
            memory_before = get_memory_usage()
            
            try:
                file_type = determine_file_type(file)
                action_type = process_file_by_type(file_type, file_path)
                
                # Update stats with correct plural forms
                if action_type == "video":
                    stats["videos"] += 1
                elif action_type == "image":
                    stats["images"] += 1
                elif action_type == "audio":
                    stats["audio"] += 1
                elif action_type == "text":
                    stats["text"] += 1
                
                print(f"✅ Successfully processed: {file_path}")
                stats["processed"] += 1
                
                # Log memory usage after processing
                current_memory = log_memory_usage(file_path, memory_before, file_index + 1, total_files, action_type, "success")
                
                # Update memory statistics
                stats["memory_stats"]["current"] = current_memory
                if current_memory > stats["memory_stats"]["peak"]:
                    stats["memory_stats"]["peak"] = current_memory
                
                # Remove from failed files if it was there (successful retry)
                if abs_path in failed_files:
                    del failed_files[abs_path]
                    save_failed_files(failed_files)
                    print(f"✅ Removed from failed list: {abs_path}")
                    
                # Enhanced memory cleanup using smart scheduling
                memory_freed = enhanced_cleanup_for_batch_processing(file_index + 1, action_type, total_files)
                if memory_freed > 0:
                    print(f"🧹 Enhanced cleanup freed: {memory_freed:.1f} MB")
                    
            except Exception as e:
                error_msg = f"Failed to process {file_path}: {e}"
                print(f"⚠️ {error_msg}")
                
                # Use helper function to determine action type for failed files
                failed_action_type = determine_file_type(file)
                
                # Log memory usage even for failed files
                current_memory = log_memory_usage(file_path, memory_before, file_index + 1, total_files, failed_action_type, "failed", error_msg)
                stats["memory_stats"]["current"] = current_memory
                if current_memory > stats["memory_stats"]["peak"]:
                    stats["memory_stats"]["peak"] = current_memory
                
                # Save the failed file with error details
                save_failed_files({abs_path: str(e)})
                stats["failed"] += 1
                stats["errors"].append(error_msg)
                
                # Continue processing other files
                continue
                            
    # Final cleanup of processor instances
    del text_processor
    gc.collect()
    
    # Final memory cleanup and statistics
    final_memory = get_memory_usage()
    stats["memory_stats"]["final"] = final_memory
    memory_growth = final_memory - stats["memory_stats"]["initial"]
    
    # Print final statistics
    print("\n" + "="*60)
    print("📊 BATCH PROCESSING COMPLETE")
    print("="*60)
    print(f"✅ Successfully processed: {stats['processed']} files")
    print(f"⏭️  Skipped (already tagged): {stats['skipped']} files")
    print(f"❌ Failed: {stats['failed']} files")
    
    print(f"\n💾 MEMORY STATISTICS")
    print(f"   Initial: {stats['memory_stats']['initial']:.1f} MB")
    print(f"   Peak: {stats['memory_stats']['peak']:.1f} MB")
    print(f"   Final: {final_memory:.1f} MB")
    print(f"   Growth: {memory_growth:+.1f} MB")
    
    # Get memory log summary
    memory_summary = get_memory_summary()
    if memory_summary:
        print(f"\n📋 MEMORY LOG SUMMARY")
        print(f"   Session started: {memory_summary['session_start']}")
        print(f"   Total entries logged: {memory_summary['total_entries']}")
        print(f"   Memory after range: {memory_summary['memory']['min_after']:.1f} - {memory_summary['memory']['max_after']:.1f} MB")
        print(f"   Total memory used: {memory_summary['memory']['total_used']:.1f} MB")
        print(f"   Average per file: {memory_summary['memory']['avg_used_per_file']:.1f} MB")
        print(f"   Largest single file: {memory_summary['memory']['max_used_single_file']:.1f} MB")
        print(f"   Actions processed: {memory_summary['actions']}")
        print(f"   Success/Failed: {memory_summary['status']['success']}/{memory_summary['status']['failed']}")
        print(f"   Memory log saved to: {get_memory_log_path()}")
    
    if memory_growth > 100:
        print(f"⚠️  SIGNIFICANT MEMORY GROWTH: {memory_growth:.1f} MB")
        print(f"   Consider investigating memory leaks or reducing batch size")
    elif memory_growth > 50:
        print(f"🔶 Moderate memory growth: {memory_growth:.1f} MB")
    
    if stats["errors"]:
        print(f"\n❌ ERRORS ENCOUNTERED:")
        for error in stats["errors"][:5]:  # Show first 5 errors
            print(f"   - {error}")
        if len(stats["errors"]) > 5:
            print(f"   ... and {len(stats['errors']) - 5} more errors")
        print(f"\n💡 Failed files are tracked in '{get_failed_files_path()}'")
        print(f"   Files will be retried up to 3 times before being permanently skipped")
    
    print("Processing complete!")
    emit_progress("directory_processing", 100, f"Directory processing complete! Processed {stats['processed']}, skipped {stats['skipped']}, failed {stats['failed']} files")
    
        # Final memory reporting
    final_memory_after_cleanup = get_memory_usage()
    total_memory_change = final_memory_after_cleanup - stats["memory_stats"]["initial"]
    print(f"📊 Final memory usage: {final_memory_after_cleanup:.1f} MB")
    print(f"📊 Net memory change from start: {total_memory_change:+.1f} MB")
        
    if total_memory_change < 100:
        print(f"✅ Memory growth successfully controlled!")
    else:
        print(f"⚠️  Memory growth: {total_memory_change:.1f} MB")
        print(f"💡 Models remain loaded for future processing")
    
    # Return stats for the unified service
    return {
        "total_files": total_files,
        "processed": stats["processed"],
        "skipped": stats["skipped"],
        "failed": stats["failed"],
        "memory_stats": stats["memory_stats"],
        "errors": stats["errors"]
    }

# Enhanced memory management function
def enhanced_cleanup_for_batch_processing(file_index, action_type, total_files):
    """Enhanced cleanup function using modular processor-specific cleanup functions"""
    from audioprocessor import cleanup_audio_processing
    memory_before = get_memory_usage()
    
    # Always cleanup after videos (they use the most memory)
    # Cleanup every file for images/audio/text, every 3 files for videos
    should_cleanup = (
        action_type == "video" or  # Always cleanup after videos
        file_index % 3 == 0 or     # Cleanup every 3 files for any type  
        file_index % 10 == 0       # Periodic cleanup every 10 files
    )
    
    if should_cleanup:
        print(f"🧹 Performing {action_type}-specific cleanup after file #{file_index}...")
        
        # Use processor-specific cleanup functions
        if action_type == "video":
            # Video cleanup is handled within videotagger.py
            pass
        elif action_type == "audio":
            cleanup_audio_processing()  # Call with no args for general cleanup

    
        # General framework cache cleanup
        for _ in range(2):
            gc.collect()
        
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
                torch.mps.empty_cache()
        except ImportError:
            pass
        
        try:
            import tensorflow as tf
            tf.keras.backend.clear_session()
        except ImportError:
            pass
    
    memory_after = get_memory_usage()
    memory_freed = memory_before - memory_after if should_cleanup else 0.0
    
    if memory_freed > 0:
        print(f"  ✅ Batch cleanup freed {memory_freed:.1f} MB ({memory_before:.1f} → {memory_after:.1f} MB)")
    
    return memory_freed

if __name__ == "__main__":
    # Handle special commands BEFORE importing heavy modules
    if len(sys.argv) > 1:
        if sys.argv[1] == "--status":
            check_status()
            sys.exit(0)
        elif sys.argv[1] == "--stop":
            stop_running_instance()
            sys.exit(0)
        elif sys.argv[1] == "--help":
            print("🎬 Content Cache Directory Processor")
            print("="*50)
            print("Usage:")
            print("  python tagdirectory.py <directory>  - Process directory")
            print("  python tagdirectory.py --status     - Check if running")
            print("  python tagdirectory.py --stop       - Stop running instance")
            print("  python tagdirectory.py --help       - Show this help")
            print("\nFeatures:")
            print("  • Only one instance can run at a time")
            print("  • New instances automatically replace old ones")
            print("  • Graceful shutdown with Ctrl+C")
            print("  • Automatic cleanup on exit")
            sys.exit(0)
    
    # Ensure only one instance is running
    ensure_single_instance()
    
    print("\n🎬 Content Cache Directory Processor")
    print("="*50)
    
    if len(sys.argv) > 1:
        directory = sys.argv[1]
    else:
        directory = input("📁 Enter path to directory: ")
    
    print(f"📂 Processing directory: {directory}")
    print(f"🔒 Process locked (PID: {get_current_pid()})")
    
    try:
        batch_process_files(directory)
    except KeyboardInterrupt:
        print(f"\n⚠️  Processing interrupted by user")
    except Exception as e:
        print(f"\n❌ Processing failed: {e}")
    finally:
        cleanup_on_exit()
        print(f"🧹 Cleanup complete")