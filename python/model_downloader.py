#!/usr/bin/env python3
"""
Model Downloader for silk.ai

Downloads all required models on first app launch with progress reporting.
Ensures the app works offline after initial setup.
"""

import os
import sys
import json
import time
from pathlib import Path
from typing import Dict, List, Callable, Optional
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ModelDownloader:
    """
    Downloads and manages ML models for silk.ai
    """
    
    def __init__(self, progress_callback: Optional[Callable] = None):
        """
        Initialize the model downloader.
        
        Args:
            progress_callback: Function to call with progress updates
                              Signature: callback(model_name, progress_percent, message)
        """
        self.progress_callback = progress_callback
        self.models_dir = self._get_models_cache_dir()
        
        # Model definitions
        self.models = {
            'sentence_transformer': {
                'name': 'SentenceTransformer (all-MiniLM-L6-v2)',
                'size_mb': 87,
                'required': True,
                'download_func': self._download_sentence_transformer
            },
            'whisper': {
                'name': 'Whisper (base)',
                'size_mb': 139,
                'required': False,  # Only needed for audio processing
                'download_func': self._download_whisper
            },
            'yamnet': {
                'name': 'YAMNet',
                'size_mb': 17,
                'required': False,  # Only needed for audio classification
                'download_func': self._download_yamnet
            }
        }
        
        logger.info(f"Model downloader initialized. Cache directory: {self.models_dir}")
    
    def _get_models_cache_dir(self) -> str:
        """Get the models cache directory."""
        # Check if running in packaged app
        if getattr(sys, 'frozen', False):
            # Running in packaged app
            app_dir = os.path.dirname(sys.executable)
            cache_dir = os.path.join(app_dir, '.silk-ai', 'models')
        else:
            # Running in development
            home_dir = os.path.expanduser('~')
            cache_dir = os.path.join(home_dir, '.silk-ai', 'models')
        
        os.makedirs(cache_dir, exist_ok=True)
        return cache_dir
    
    def _emit_progress(self, model_name: str, progress: float, message: str):
        """Emit progress update if callback is provided."""
        if self.progress_callback:
            self.progress_callback(model_name, progress, message)
        logger.info(f"[{model_name}] {progress:.1f}% - {message}")
    
    def _download_sentence_transformer(self) -> bool:
        """Download SentenceTransformer model."""
        try:
            self._emit_progress('sentence_transformer', 0, 'Initializing SentenceTransformer download...')
            
            from sentence_transformers import SentenceTransformer
            
            # Download with progress tracking
            model = SentenceTransformer(
                'all-MiniLM-L6-v2',
                cache_folder=self.models_dir
            )
            
            self._emit_progress('sentence_transformer', 100, 'SentenceTransformer downloaded successfully')
            return True
            
        except Exception as e:
            self._emit_progress('sentence_transformer', 0, f'Failed to download SentenceTransformer: {e}')
            logger.error(f"SentenceTransformer download failed: {e}")
            return False
    
    def _download_whisper(self) -> bool:
        """Download Whisper model."""
        try:
            self._emit_progress('whisper', 0, 'Initializing Whisper download...')
            
            import whisper
            
            # Download Whisper model
            model = whisper.load_model("base", download_root=self.models_dir)
            
            self._emit_progress('whisper', 100, 'Whisper downloaded successfully')
            return True
            
        except Exception as e:
            self._emit_progress('whisper', 0, f'Failed to download Whisper: {e}')
            logger.error(f"Whisper download failed: {e}")
            return False
    
    def _download_yamnet(self) -> bool:
        """Download YAMNet model."""
        try:
            self._emit_progress('yamnet', 0, 'Initializing YAMNet download...')
            
            import tensorflow_hub as hub
            
            # Set TensorFlow Hub cache directory
            os.environ['TFHUB_CACHE_DIR'] = os.path.join(self.models_dir, 'tfhub')
            Path(os.environ['TFHUB_CACHE_DIR']).mkdir(exist_ok=True)
            
            # Download YAMNet model
            model = hub.load("https://tfhub.dev/google/yamnet/1")
            
            self._emit_progress('yamnet', 100, 'YAMNet downloaded successfully')
            return True
            
        except Exception as e:
            self._emit_progress('yamnet', 0, f'Failed to download YAMNet: {e}')
            logger.error(f"YAMNet download failed: {e}")
            return False
    
    def check_models_status(self) -> Dict[str, bool]:
        """
        Check which models are already downloaded.
        
        Returns:
            Dictionary mapping model names to download status
        """
        status = {}
        
        # Check SentenceTransformer
        sentence_path = os.path.join(self.models_dir, "models--sentence-transformers--all-MiniLM-L6-v2")
        status['sentence_transformer'] = os.path.exists(sentence_path)
        
        # Check Whisper
        whisper_path = os.path.join(self.models_dir, "base.pt")
        status['whisper'] = os.path.exists(whisper_path)
        
        # Check YAMNet
        yamnet_path = os.path.join(self.models_dir, "tfhub")
        status['yamnet'] = os.path.exists(yamnet_path) and len(os.listdir(yamnet_path)) > 0
        
        return status
    
    def download_required_models(self) -> bool:
        """
        Download only required models (SentenceTransformer).
        
        Returns:
            True if all required models downloaded successfully
        """
        status = self.check_models_status()
        
        # Only download required models that are missing
        required_models = [name for name, config in self.models.items() 
                          if config['required'] and not status.get(name, False)]
        
        if not required_models:
            self._emit_progress('all', 100, 'All required models already downloaded')
            return True
        
        success = True
        for i, model_name in enumerate(required_models):
            model_config = self.models[model_name]
            
            self._emit_progress('all', (i / len(required_models)) * 100, 
                              f'Downloading {model_config["name"]}...')
            
            if not model_config['download_func']():
                success = False
                break
        
        if success:
            self._emit_progress('all', 100, 'All required models downloaded successfully')
        else:
            self._emit_progress('all', 0, 'Failed to download some required models')
        
        return success
    
    def is_offline_ready(self) -> bool:
        """
        Check if the app can run offline (required models are downloaded).
        
        Returns:
            True if app can run offline
        """
        status = self.check_models_status()
        required_models = [name for name, config in self.models.items() if config['required']]
        
        return all(status.get(model_name, False) for model_name in required_models)


def main():
    """CLI interface for model downloader."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Download models for silk.ai")
    parser.add_argument("--required-only", action="store_true", 
                       help="Download only required models")
    parser.add_argument("--status", action="store_true", 
                       help="Show download status")
    
    args = parser.parse_args()
    
    def progress_callback(model_name: str, progress: float, message: str):
        print(f"[{model_name}] {progress:.1f}% - {message}")
    
    downloader = ModelDownloader(progress_callback)
    
    if args.status:
        status = downloader.check_models_status()
        print(f"\n📋 Model Status:")
        for model_name, downloaded in status.items():
            status_icon = '✅' if downloaded else '❌'
            model_info = downloader.models[model_name]
            print(f"  {status_icon} {model_info['name']} ({model_info['size_mb']}MB)")
        print(f"\nOffline ready: {'✅' if downloader.is_offline_ready() else '❌'}")
            
    elif args.required_only:
        success = downloader.download_required_models()
        if success:
            print("✅ All required models downloaded successfully")
        else:
            print("❌ Failed to download required models")
            sys.exit(1)
    else:
        print("Use --required-only to download models or --status to check status")


if __name__ == "__main__":
    main() 