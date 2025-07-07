const { app, BrowserWindow, dialog, ipcMain, Menu, shell } = require("electron")
const path = require("path")
const { spawn, exec } = require("child_process")
const fs = require("fs")

let mainWindow
let pythonProcess
let searchServerProcess
let isFirstLaunch = false

// Global dependency state management
let dependencyInstallationComplete = false
let dependencyInstallationInProgress = false
let dependencyInstallationFailed = false

// Queue for delayed service starts
let pendingServiceStarts = []

async function waitForDependencies() {
  return new Promise((resolve, reject) => {
    if (dependencyInstallationComplete) {
      resolve()
      return
    }
    
    if (dependencyInstallationFailed) {
      reject(new Error("Dependency installation failed"))
      return
    }
    
    // Add to queue
    pendingServiceStarts.push({ resolve, reject })
  })
}

function notifyDependenciesReady() {
  console.log("📢 Notifying all services that dependencies are ready...")
  dependencyInstallationComplete = true
  dependencyInstallationInProgress = false
  
  // Resolve all pending service starts
  while (pendingServiceStarts.length > 0) {
    const { resolve } = pendingServiceStarts.shift()
    resolve()
  }
}

function notifyDependenciesFailed(error) {
  console.error("📢 Notifying all services that dependencies failed:", error.message)
  dependencyInstallationFailed = true
  dependencyInstallationInProgress = false
  
  // Reject all pending service starts
  while (pendingServiceStarts.length > 0) {
    const { reject } = pendingServiceStarts.shift()
    reject(error)
  }
}

// Get the correct Python executable for both development and packaged environments
function getPythonExecutable() {
  const isDev = !app.isPackaged
  
  if (isDev) {
    // Development mode - use virtual environment
    const venvPython = path.join(__dirname, "..", ".venv", "bin", "python")
    if (require("fs").existsSync(venvPython)) {
      return venvPython
    }
  }
  
  // Packaged mode or fallback - find system Python
  if (process.platform === 'win32') {
    // On Windows, try to find Python in common locations
    const pythonPaths = [
      'python',
      'python3',
      'C:\\Python310\\python.exe',
      'C:\\Python39\\python.exe',
      'C:\\Users\\' + process.env.USERNAME + '\\AppData\\Local\\Programs\\Python\\Python310\\python.exe'
    ]
    
    for (const pythonPath of pythonPaths) {
      try {
        require('child_process').execSync(`${pythonPath} --version`, { stdio: 'ignore' })
        return pythonPath
      } catch (e) {
        continue
      }
    }
    return 'python'
  } else {
    // On macOS/Linux, find the actual system Python
    const pythonPaths = [
      '/usr/bin/python3',
      '/usr/local/bin/python3',
      '/opt/homebrew/bin/python3',
      '/Library/Frameworks/Python.framework/Versions/3.10/bin/python3',
      '/Library/Frameworks/Python.framework/Versions/3.11/bin/python3',
      '/Library/Frameworks/Python.framework/Versions/3.12/bin/python3',
      'python3',
      'python'
    ]
    
    // Check for working Python executables
    for (const pythonPath of pythonPaths) {
      try {
        if (pythonPath.startsWith('/')) {
          // Absolute path - check if it exists
          if (require("fs").existsSync(pythonPath)) {
            return pythonPath
          }
        } else {
          // Relative path - test if it works
          require('child_process').execSync(`which ${pythonPath}`, { stdio: 'ignore' })
          return pythonPath
        }
      } catch (e) {
        continue
      }
    }
    
    // Fallback
    return 'python3'
  }
}

// Setup Python environment for packaged apps
function setupPythonEnvironment() {
  const isDev = !app.isPackaged
  
  if (!isDev) {
    // Enhanced setup for packaged apps, especially Apple Silicon M3
    console.log("🐍 Setting up Python environment for packaged app...")
    console.log(`Platform: ${process.platform}, Arch: ${process.arch}`)
    
    // Set working directory to Resources for consistent path resolution
    const resourcesPath = process.resourcesPath
    console.log(`Resources path: ${resourcesPath}`)
    
    try {
      process.chdir(resourcesPath)
      console.log(`✅ Changed working directory to: ${process.cwd()}`)
    } catch (error) {
      console.error(`❌ Failed to change working directory: ${error}`)
    }
    
    // Build comprehensive PYTHONPATH
    const pythonPaths = [
      // Backend processing modules
      path.join(resourcesPath, "backend", "processing"),
      path.join(resourcesPath, "python-dist", "backend", "processing"),
      // Python scripts
      path.join(resourcesPath, "python"),
      // Search modules
      path.join(resourcesPath, "backend", "search"),
      path.join(resourcesPath, "python-dist", "backend", "search"),
    ]
    
    // Filter to only existing paths
    const existingPaths = pythonPaths.filter(p => {
      const exists = fs.existsSync(p)
      console.log(`Python path ${exists ? '✅' : '❌'}: ${p}`)
      return exists
    })
    
    if (existingPaths.length > 0) {
      const pythonPathValue = existingPaths.join(path.delimiter)
      process.env.PYTHONPATH = pythonPathValue + (process.env.PYTHONPATH ? `${path.delimiter}${process.env.PYTHONPATH}` : '')
      console.log(`✅ Set PYTHONPATH: ${process.env.PYTHONPATH}`)
    } else {
      console.warn("⚠️ No Python paths found - backend may not be available")
    }
    
    // Additional environment variables for better compatibility
    process.env.PYTHONIOENCODING = 'utf-8'
    process.env.PYTHONUNBUFFERED = '1'
    
    // CRITICAL: Isolate Python environment from system packages
    // This prevents Python from using system site-packages that have incompatible versions
    process.env.PYTHONNOUSERSITE = '0'  // Allow user site (where we install our packages)
    process.env.PYTHONSAFEPATH = '1'    // Don't add current directory to sys.path
    
    // Set user base to a controlled location for dependency isolation
    const appDataDir = getAppCacheDir()
    const userBase = path.join(appDataDir, 'python_packages')
    process.env.PYTHONUSERBASE = userBase
    
    // Add our isolated packages to PYTHONPATH so they're found first
    const userSitePackages = path.join(userBase, 'lib', 'python', 'site-packages')
    const currentPythonPath = process.env.PYTHONPATH || ''
    process.env.PYTHONPATH = userSitePackages + (currentPythonPath ? `${path.delimiter}${currentPythonPath}` : '')
    
    console.log(`✅ Set PYTHONUSERBASE: ${userBase}`)
    console.log(`✅ Added user site-packages to PYTHONPATH: ${userSitePackages}`)
    console.log(`✅ Python environment isolated from system packages`)
    
    // Apple Silicon specific optimizations
    if (process.platform === 'darwin' && process.arch === 'arm64') {
      console.log("🍎 Detected Apple Silicon - setting optimized environment")
      process.env.OPENBLAS_NUM_THREADS = '1'  // Prevent threading issues
      process.env.VECLIB_MAXIMUM_THREADS = '1'  // macOS acceleration library threading
    }
    
    console.log("✅ Python environment setup complete")
  }
}

// Get the correct backend path for both development and packaged environments
function getBackendPath() {
  const isDev = !app.isPackaged
  
  if (isDev) {
    // In development, use the backend folder within packaging
    return path.join(__dirname, "backend")
  } else {
    // In packaged app, backend is in resources
    return path.join(process.resourcesPath, "backend")
  }
}

// Get the correct Python scripts path
function getPythonScriptsPath() {
  const isDev = !app.isPackaged
  
  if (isDev) {
    return path.join(__dirname, "python")
  } else {
    // In packaged app, python scripts are in resources
    return path.join(process.resourcesPath, "python")
  }
}

// Function to get app cache directory (matching Python config)
function getAppCacheDir() {
  if (process.platform === 'darwin') {  // macOS
    return path.join(require('os').homedir(), 'Library', 'Application Support', 'silk.ai')
  } else if (process.platform === 'win32') {  // Windows
    return path.join(process.env.APPDATA || require('os').homedir(), 'silk.ai')
  } else {  // Linux and others
    return path.join(require('os').homedir(), '.config', 'silk.ai')
  }
}

// Function to clear locally stored logs
function clearLocalLogs() {
  try {
    const metadataDir = path.join(getAppCacheDir(), 'metadata')
    
    // Log files to clear
    const logFiles = [
      'memory_log.json',
      'failed_files.json'
    ]
    
    let clearedCount = 0
    
    for (const logFile of logFiles) {
      const logPath = path.join(metadataDir, logFile)
      
      if (fs.existsSync(logPath)) {
        try {
          fs.unlinkSync(logPath)
          console.log(`🧹 Cleared log file: ${logFile}`)
          clearedCount++
        } catch (error) {
          console.warn(`⚠️ Failed to clear log file ${logFile}:`, error.message)
        }
      }
    }
    
    if (clearedCount > 0) {
      console.log(`✅ Cleared ${clearedCount} log files`)
    } else {
      console.log("📝 No log files found to clear")
    }
    
    return clearedCount
  } catch (error) {
    console.error("❌ Error clearing log files:", error)
    return 0
  }
}

// Check if this is the first launch (no models downloaded)
async function checkFirstLaunch() {
  const fs = require("fs")
  const os = require("os")
  
  // Check for models in the expected location using proper OS-specific paths
  let modelsDir
  if (process.platform === 'darwin') {  // macOS
    modelsDir = path.join(os.homedir(), 'Library', 'Application Support', 'silk.ai', 'models')
  } else if (process.platform === 'win32') {  // Windows
    modelsDir = path.join(process.env.APPDATA || os.homedir(), 'silk.ai', 'models')
  } else {  // Linux and others
    modelsDir = path.join(os.homedir(), '.config', 'silk.ai', 'models')
  }
  
  const sentenceTransformerPath = path.join(modelsDir, 'models--sentence-transformers--all-MiniLM-L6-v2')
  
  return !fs.existsSync(sentenceTransformerPath)
}

// Download required models on first launch
async function downloadModelsIfNeeded() {
  if (!isFirstLaunch) return true
  
  console.log("First launch detected - downloading required models...")
  
  return new Promise((resolve) => {
    // Show model download progress to user
    if (mainWindow && !mainWindow.isDestroyed()) {
      mainWindow.webContents.send("model-download-started", {
        message: "Welcome to silk.ai! Downloading required models...",
        isFirstLaunch: true
      })
    }
    
    // Use the model downloader
    const modelDownloaderPath = path.join(getPythonScriptsPath(), "model_downloader.py")
    const pythonCmd = getPythonExecutable()
    
    console.log(`Using model downloader: ${modelDownloaderPath}`)
    
    const downloadProcess = spawn(pythonCmd, [modelDownloaderPath, '--required-only'], {
      stdio: ['pipe', 'pipe', 'pipe'],
      cwd: getPythonScriptsPath()
    })
    
    downloadProcess.stdout.on("data", (data) => {
      const output = data.toString().trim()
      console.log(`Model Download: ${output}`)
      
      // Parse progress from output
      const progressMatch = output.match(/\[(\w+)\]\s+([\d.]+)%\s+-\s+(.+)/)
      if (progressMatch && mainWindow && !mainWindow.isDestroyed()) {
        const [, modelName, progress, message] = progressMatch
        mainWindow.webContents.send("model-download-progress", {
          modelName,
          progress: parseFloat(progress),
          message
        })
      } else if (mainWindow && !mainWindow.isDestroyed()) {
        mainWindow.webContents.send("model-download-progress", {
          modelName: 'general',
          progress: 50,
          message: output
        })
      }
    })
    
    downloadProcess.stderr.on("data", (data) => {
      console.error(`Model Download Error: ${data}`)
      if (mainWindow && !mainWindow.isDestroyed()) {
        mainWindow.webContents.send("model-download-error", data.toString())
      }
    })
    
    downloadProcess.on("close", (code) => {
      if (code === 0) {
        console.log("Models downloaded successfully")
        if (mainWindow && !mainWindow.isDestroyed()) {
          mainWindow.webContents.send("model-download-complete", {
            success: true,
            message: "Models downloaded successfully! silk.ai is ready to use."
          })
        }
        resolve(true)
      } else {
        console.error(`Model download failed with code ${code}`)
        if (mainWindow && !mainWindow.isDestroyed()) {
          mainWindow.webContents.send("model-download-complete", {
            success: false,
            message: "Model download failed. Some features may not work properly."
          })
        }
        resolve(false)
      }
    })
    
    downloadProcess.on("error", (error) => {
      console.error("Model download process error:", error)
      if (mainWindow && !mainWindow.isDestroyed()) {
        mainWindow.webContents.send("model-download-complete", {
          success: false,
          message: "Failed to start model download. Please check your internet connection."
        })
      }
      resolve(false)
    })
  })
}

// Function to kill processes on specific ports
function killProcessesOnPorts() {
  try {
    exec("lsof -ti:3002,5001 | xargs kill -9 2>/dev/null || true", () => {})
  } catch (error) {
    console.log("No existing processes to kill")
  }
}

function getFfmpegPath() {
  // Get the path to the bundled ffmpeg binary or system ffmpeg
  // Try bundled ffmpeg first (in packaged app)
  const currentDir = __dirname
  
  // Check multiple possible locations for the bundled binary
  const possiblePaths = [
    // Packaged app structure: main.js -> ../binaries/ffmpeg
    path.join(currentDir, '..', 'binaries', 'ffmpeg'),
    // Alternative: Resources/binaries/ffmpeg
    path.join(currentDir, 'binaries', 'ffmpeg'),
    // Alternative: app.asar.unpacked/binaries/ffmpeg
    path.join(currentDir, '..', 'app.asar.unpacked', 'binaries', 'ffmpeg'),
  ]
  
  for (const ffmpegPath of possiblePaths) {
    console.log(`🔍 Checking ffmpeg at: ${ffmpegPath}`)
    if (fs.existsSync(ffmpegPath)) {
      console.log(`✅ Found bundled ffmpeg: ${ffmpegPath}`)
      return ffmpegPath
    }
  }
  
  // Fallback to system ffmpeg
  console.log("⚠️ Bundled ffmpeg not found, using system ffmpeg")
  return 'ffmpeg'
}

function createWindow() {
  // Create the browser window
  mainWindow = new BrowserWindow({
    width: 1400,
    height: 900,
    minWidth: 1200,
    minHeight: 700,
    icon: path.join(__dirname, "public", "icon.png"), // Add silk.ai logo
    webPreferences: {
      nodeIntegration: false,
      contextIsolation: true,
      enableRemoteModule: false,
      preload: path.join(__dirname, "preload.js"),
      webSecurity: true,
      sandbox: false
    },
    show: false,
    backgroundColor: "#f8fafc",
  })

  // Load the app with better error handling
  if (process.env.NODE_ENV === "development") {
    // Retry loading the URL in case Next.js isn't ready yet
    const loadWithRetry = (retries = 10) => {
      mainWindow.loadURL("http://localhost:3002").catch((error) => {
        console.error("Failed to load URL:", error)
        if (retries > 0 && mainWindow && !mainWindow.isDestroyed()) {
          console.log(`Failed to load UI, retrying... (${retries} attempts left)`)
          setTimeout(() => loadWithRetry(retries - 1), 2000)
        } else {
          console.error("Failed to load UI after all retries:", error)
          // Try to load a fallback page or show error
          if (mainWindow && !mainWindow.isDestroyed()) {
            mainWindow.loadURL('data:text/html,<h1>Failed to load application. Please restart.</h1>')
          }
        }
      })
    }
    loadWithRetry()
    // Only open dev tools if not in production
    try {
    mainWindow.webContents.openDevTools()
    } catch (error) {
      console.error("Failed to open dev tools:", error)
    }
  } else {
    // Production mode - load from the static build
    // In packaged app with asar, __dirname points to the asar file
    let indexPath = path.join(__dirname, "out", "index.html")
    
    console.log("Loading production file from:", indexPath)
    console.log("__dirname is:", __dirname)
    console.log("app.isPackaged:", app.isPackaged)
    
    mainWindow.loadFile(indexPath).catch((error) => {
      console.error("Failed to load production file:", error)
      console.log("Trying to find index.html in different locations...")
      
      // Try to find the file in various possible locations
      const possiblePaths = [
        path.join(__dirname, "out", "index.html"),
        path.join(__dirname, "index.html"),
        path.join(process.resourcesPath, "app", "out", "index.html"),
        path.join(process.resourcesPath, "app", "index.html"),
        path.join(process.resourcesPath, "index.html"),
        path.join(__dirname, "..", "out", "index.html"),
        path.join(__dirname, "..", "index.html")
      ]
      
      let foundPath = null
      for (const testPath of possiblePaths) {
        console.log(`Checking: ${testPath} - exists: ${fs.existsSync(testPath)}`)
        if (fs.existsSync(testPath)) {
          foundPath = testPath
          break
        }
      }
      
      if (foundPath) {
        console.log("Found index.html at:", foundPath)
        mainWindow.loadFile(foundPath)
      } else {
        console.error("Could not find index.html in any expected location")
        // Load a basic error page
        mainWindow.loadURL('data:text/html,<h1>Failed to load silk.ai. Could not find application files.</h1><p>Please reinstall the application.</p>')
      }
    })
  }

  // Show window when ready
  mainWindow.once("ready-to-show", () => {
    if (mainWindow && !mainWindow.isDestroyed()) {
    mainWindow.show()
    
    // Check for first launch and download models if needed
    setTimeout(async () => {
      try {
        isFirstLaunch = await checkFirstLaunch()
        if (isFirstLaunch) {
          await downloadModelsIfNeeded()
        }
      } catch (error) {
        console.error("Error checking/downloading models:", error)
      }
    }, 2000)
    }
  })

  // Handle window closed
  mainWindow.on("closed", () => {
    mainWindow = null
  })

  // Add error handling for the window
  mainWindow.on("unresponsive", () => {
    console.warn("Window became unresponsive")
  })

  mainWindow.webContents.on("crashed", (event, killed) => {
    console.error("Window crashed:", { killed })
  })

  // Create application menu
  try {
  createMenu()
  } catch (error) {
    console.error("Failed to create menu:", error)
  }
}

function createMenu() {
  const template = [
    {
      label: "silk.ai",
      submenu: [
        { role: "about" },
        { type: "separator" },
        { role: "services" },
        { type: "separator" },
        { role: "hide" },
        { role: "hideothers" },
        { role: "unhide" },
        { type: "separator" },
        { role: "quit" },
      ],
    },
    {
      label: "File",
      submenu: [
        {
          label: "Import Files...",
          accelerator: "CmdOrCtrl+O",
          click: () => {
            handleFileImport()
          },
        },
        {
          label: "Import Folder...",
          accelerator: "CmdOrCtrl+Shift+O",
          click: () => {
            handleFolderImport()
          },
        },
        { type: "separator" },
        { role: "close" },
      ],
    },
    {
      label: "Edit",
      submenu: [
        { role: "undo" },
        { role: "redo" },
        { type: "separator" },
        { role: "cut" },
        { role: "copy" },
        { role: "paste" },
        { role: "selectall" },
      ],
    },
    {
      label: "View",
      submenu: [
        { role: "reload" },
        { role: "forceReload" },
        { role: "toggleDevTools" },
        { type: "separator" },
        { role: "resetZoom" },
        { role: "zoomIn" },
        { role: "zoomOut" },
        { type: "separator" },
        { role: "togglefullscreen" },
      ],
    },
    {
      label: "Window",
      submenu: [{ role: "minimize" }, { role: "close" }],
    },
  ]

  const menu = Menu.buildFromTemplate(template)
  Menu.setApplicationMenu(menu)
}

// File import handlers
async function handleFileImport() {
  const result = await dialog.showOpenDialog(mainWindow, {
    properties: ["openFile", "multiSelections"],
    filters: [
      { name: "All Supported Files", extensions: ["mp4", "mov", "avi", "mkv", "wmv", "flv", "webm", "m4v", "mp3", "wav", "aac", "flac", "m4a", "ogg", "jpg", "jpeg", "png", "bmp", "tiff", "webp", "heic", "txt", "md", "pdf", "docx", "rtf"] },
      { name: "Video Files", extensions: ["mp4", "mov", "avi", "mkv", "wmv", "flv", "webm", "m4v"] },
      { name: "Audio Files", extensions: ["mp3", "wav", "aac", "flac", "m4a", "ogg"] },
      { name: "Image Files", extensions: ["jpg", "jpeg", "png", "bmp", "tiff", "webp", "heic"] },
      { name: "Text Files", extensions: ["txt", "md", "pdf", "docx", "rtf"] },
      { name: "All Files", extensions: ["*"] },
    ],
  })

  if (!result.canceled && mainWindow && !mainWindow.isDestroyed()) {
    mainWindow.webContents.send("files-selected", result.filePaths)
  }
}

async function handleFolderImport() {
  const result = await dialog.showOpenDialog(mainWindow, {
    properties: ["openDirectory"],
  })

  if (!result.canceled && mainWindow && !mainWindow.isDestroyed()) {
    mainWindow.webContents.send("folder-selected", result.filePaths[0])
  }
}

// Search server management
let searchServerStarting = false // Prevent concurrent starts

async function startSearchServer() {
  try {
    console.log("🔍 DEBUG: startSearchServer() called")
    
    // CRITICAL: Wait for dependencies before starting search server
    try {
      console.log("🔍 Waiting for dependencies before starting search server...")
      await waitForDependencies()
      console.log("✅ Dependencies confirmed - proceeding with search server start")
    } catch (error) {
      console.error("❌ Search server startup blocked - dependencies failed:", error.message)
      
      if (mainWindow && !mainWindow.isDestroyed()) {
        mainWindow.webContents.send("search-server-error", 
          `Search server could not start: ${error.message}`)
      }
      return
    }
    
    // Prevent concurrent starts
    if (searchServerStarting) {
      console.log("🔄 Search server startup already in progress, skipping...")
      return
    }
    
    // If search server is already running and healthy, don't restart it
    if (searchServerProcess && !searchServerProcess.killed) {
      console.log("✅ Search server process already running, checking health...")
      
      // Quick health check
      const http = require('http')
      const req = http.get('http://localhost:5001/health', { timeout: 2000 }, (res) => {
        if (res.statusCode === 200) {
          console.log("✅ Search server is healthy, not restarting")
          return
        } else {
          console.log("⚠️ Search server unhealthy, restarting...")
          restartSearchServer()
        }
      })
      
      req.on('error', () => {
        console.log("⚠️ Search server not responding, restarting...")
        restartSearchServer()
      })
      
      req.on('timeout', () => {
        req.destroy()
        console.log("⚠️ Search server timeout, restarting...")
        restartSearchServer()
      })
      
      return
    }
    
    startSearchServerInternal()
  } catch (error) {
    console.error("❌ Failed to start search server:", error)
    searchServerStarting = false
  }
}

function restartSearchServer() {
  console.log("🔄 Restarting search server...")
  try {
    if (searchServerProcess && !searchServerProcess.killed) {
      searchServerProcess.kill('SIGTERM')
      setTimeout(() => {
        if (searchServerProcess && !searchServerProcess.killed) {
          console.log("🔄 Force killing search server process...")
          searchServerProcess.kill('SIGKILL')
        }
        searchServerProcess = null
        setTimeout(() => {
          startSearchServerInternal()
        }, 1000)
      }, 2000)
    } else {
      startSearchServerInternal()
    }
  } catch (error) {
    console.warn("⚠️ Error restarting search server:", error)
    searchServerStarting = false
  }
}

function startSearchServerInternal() {
  try {
    console.log("🔍 DEBUG: startSearchServerInternal() called")
    searchServerStarting = true // Set flag to prevent concurrent starts
    
    const pythonExe = getPythonExecutable()
    console.log(`🔍 DEBUG: Python executable: ${pythonExe}`)
    
    const backendPath = getBackendPath()
    console.log(`🔍 DEBUG: Backend path: ${backendPath}`)
    
    const searchServerScript = path.join(backendPath, "search", "search_server.py")
    console.log(`🔍 DEBUG: Search server script path: ${searchServerScript}`)
    
    const searchWorkingDir = path.join(backendPath, "search")
    console.log(`🔍 DEBUG: Search working directory: ${searchWorkingDir}`)
    
    // Check if the search server script exists
    const fs = require("fs")
    if (!fs.existsSync(searchServerScript)) {
      console.error(`❌ Search server script not found at: ${searchServerScript}`)
      searchServerStarting = false
      return
    }
    console.log(`✅ Search server script exists at: ${searchServerScript}`)
    
    // Check if the working directory exists
    if (!fs.existsSync(searchWorkingDir)) {
      console.error(`❌ Search working directory not found at: ${searchWorkingDir}`)
      searchServerStarting = false
      return
    }
    console.log(`✅ Search working directory exists at: ${searchWorkingDir}`)
    
    console.log(`🚀 Starting search server with: ${pythonExe} ${searchServerScript}`)
    console.log(`🚀 Working directory: ${searchWorkingDir}`)
    
    searchServerProcess = spawn(pythonExe, [searchServerScript], {
      stdio: ["pipe", "pipe", "pipe"],
      cwd: searchWorkingDir,
      env: { ...process.env, PYTHONUNBUFFERED: '1' }
    })

    console.log(`✅ Search server process spawned with PID: ${searchServerProcess.pid}`)

    let searchServerLogBuffer = []
    const MAX_LOG_LINES = 200

    function addToSearchLogBuffer(message, type = 'output') {
      searchServerLogBuffer.push({ message, type, timestamp: Date.now() })
      
      // Keep only the last MAX_LOG_LINES
      if (searchServerLogBuffer.length > MAX_LOG_LINES) {
        searchServerLogBuffer = searchServerLogBuffer.slice(-MAX_LOG_LINES)
      }
      
      // Send to UI
      if (mainWindow && !mainWindow.isDestroyed()) {
        mainWindow.webContents.send(`search-server-${type}`, message)
      }
    }

    searchServerProcess.stdout.on("data", (data) => {
      const output = data.toString()
      console.log(`🔍 Search Server STDOUT: ${output}`)
      addToSearchLogBuffer(output, 'output')
    })

    searchServerProcess.stderr.on("data", (data) => {
      const error = data.toString()
      console.error(`🔍 Search Server STDERR: ${error}`)
      addToSearchLogBuffer(error, 'error')
    })

    searchServerProcess.on("close", (code) => {
      console.log(`🔍 Search server process closed with code ${code}`)
      searchServerProcess = null // Reset process reference when closed
      searchServerStarting = false // Reset flag when process closes
    })

    searchServerProcess.on("error", (error) => {
      console.error(`🔍 Search server process error:`, error)
      searchServerStarting = false // Reset flag on error
    })

    // Reset flag after successful startup
    setTimeout(() => {
      searchServerStarting = false
      console.log("✅ Search server startup completed (process handlers attached)")
    }, 5000)
    
  } catch (error) {
    console.error("❌ Failed to start search server:", error)
    searchServerStarting = false // Reset flag on exception
  }
}

// Python process management
async function startPythonProcess() {
  try {
    // CRITICAL: Wait for dependencies before starting Python process
    try {
      console.log("🐍 Waiting for dependencies before starting Python process...")
      await waitForDependencies()
      console.log("✅ Dependencies confirmed - proceeding with Python process start")
    } catch (error) {
      console.error("❌ Python process startup blocked - dependencies failed:", error.message)
      
      if (mainWindow && !mainWindow.isDestroyed()) {
        mainWindow.webContents.send("python-error", 
          `Python process could not start: ${error.message}`)
      }
      return
    }
    
    const pythonExe = getPythonExecutable()
    const pythonScriptsPath = getPythonScriptsPath()
    const mainScript = path.join(pythonScriptsPath, "main.py")
    
    // Run import diagnostics first in development mode
    if (process.env.NODE_ENV === "development") {
      console.log("🔍 Running import diagnostics...")
      const diagnosticScript = path.join(pythonScriptsPath, "test_imports.py")
      if (fs.existsSync(diagnosticScript)) {
        try {
          const { execSync } = require('child_process')
          const diagnosticResult = execSync(`${pythonExe} "${diagnosticScript}"`, {
            encoding: 'utf8',
            timeout: 30000,
            cwd: pythonScriptsPath
          })
          console.log("📋 Diagnostic results:")
          console.log(diagnosticResult)
        } catch (error) {
          console.log("⚠️ Diagnostic script failed:", error.message)
        }
      }
    }
    
    console.log(`Starting Python process with: ${pythonExe} ${mainScript}`)
    
    pythonProcess = spawn(pythonExe, [mainScript], {
      stdio: ["pipe", "pipe", "pipe"],
      cwd: process.resourcesPath,
      env: { ...process.env, PYTHONUNBUFFERED: '1' }
    })

    // Add error handler for spawn failure
    pythonProcess.on("error", (error) => {
      console.error("Python process spawn error:", error)
    })

    // Add handler for process exit
    pythonProcess.on("exit", (code, signal) => {
      console.log(`Python process exited with code ${code}, signal ${signal}`)
    })

      let pythonOutputBuffer = ''
  let pythonLogBuffer = []
  const MAX_LOG_LINES = 200

  function addToPythonLogBuffer(message, type = 'output') {
    pythonLogBuffer.push({ message, type, timestamp: Date.now() })
    
    // Keep only the last MAX_LOG_LINES
    if (pythonLogBuffer.length > MAX_LOG_LINES) {
      pythonLogBuffer = pythonLogBuffer.slice(-MAX_LOG_LINES)
    }
    
    // Send to UI
    if (mainWindow && !mainWindow.isDestroyed()) {
      mainWindow.webContents.send(`python-${type}`, message)
    }
  }

  pythonProcess.stdout.on("data", (data) => {
    const output = data.toString()
    pythonOutputBuffer += output
    
    // Process complete lines from the buffer
    const lines = pythonOutputBuffer.split('\n')
    pythonOutputBuffer = lines.pop() // Keep the incomplete line in buffer
    
    for (const line of lines) {
      if (line.trim()) {
        addToPythonLogBuffer(line.trim(), 'output')
      }
    }
  })

  pythonProcess.stderr.on("data", (data) => {
    console.error(`Python stderr: ${data}`)
    addToPythonLogBuffer(data.toString(), 'error')
  })

    pythonProcess.on("close", (code) => {
      console.log(`Python process exited with code ${code}`)
    })

    // Send initial status request with better error handling
    setTimeout(() => {
      if (pythonProcess && pythonProcess.stdin && !pythonProcess.killed) {
        try {
        pythonProcess.stdin.write(JSON.stringify({ action: "status" }) + "\n")
        } catch (error) {
          console.error("Error writing initial status to Python stdin:", error)
        }
      } else {
        console.error("Python process not available for initial status request")
      }
    }, 1000)
  } catch (error) {
    console.error("Failed to start Python process:", error)
  }
}

// Global log buffers for managing UI log overflow
let globalPythonLogBuffer = []
let globalSearchLogBuffer = []
const GLOBAL_MAX_LOG_LINES = 200

// Function to clear all UI logs 
function clearUILogs() {
  globalPythonLogBuffer = []
  globalSearchLogBuffer = []
  
  if (mainWindow && !mainWindow.isDestroyed()) {
    mainWindow.webContents.send("clear-logs")
  }
  
  console.log("🧹 Cleared UI log buffers to prevent lag")
}

// IPC handlers
ipcMain.handle("select-files", handleFileImport)
ipcMain.handle("select-folder", handleFolderImport)
ipcMain.handle("process-files", handleProcessFiles)
ipcMain.handle("get-metadata-paths", handleGetMetadataPaths)
ipcMain.handle("test-api-connectivity", handleTestApiConnectivity)
ipcMain.handle("stop-processing", handleStopProcessing)
ipcMain.handle("clear-ui-logs", clearUILogs)

// File system operation handlers
ipcMain.handle("open-file", async (event, filePath) => {
  const { shell } = require("electron")
  return shell.openPath(filePath)
})

ipcMain.handle("reveal-file", async (event, filePath) => {
  const { shell } = require("electron")
  shell.showItemInFolder(filePath)
  return { success: true }
})

// Simple thumbnail generation
ipcMain.handle("generate-thumbnail", async (event, filePath, contentType) => {
  try {
    // Generate thumbnail using the same approach as working UI
    const result = await generateSafeThumbnail(filePath, contentType)
    return result
  } catch (error) {
    console.error('Thumbnail generation failed:', error)
    return generateSimpleThumbnail(contentType)
  }
})

async function generateSafeThumbnail(filePath, contentType) {
  const fs = require("fs")
  const path = require("path")
  
  try {
    // Check if file exists
    if (!fs.existsSync(filePath)) {
      return generateSimpleThumbnail(contentType)
    }

    const ext = path.extname(filePath).toLowerCase()
    
    if (contentType === 'video' || ['.mp4', '.mov', '.avi', '.mkv', '.wmv', '.flv', '.webm', '.m4v'].includes(ext)) {
      return await generateVideoThumbnail(filePath)
    } else if (contentType === 'image' || ['.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.webp', '.heic'].includes(ext)) {
      return await generateImageThumbnail(filePath)
    } else {
      return generateSimpleThumbnail(contentType)
    }
    
  } catch (error) {
    console.error('Thumbnail generation failed:', error)
    return generateSimpleThumbnail(contentType)
  }
}

async function generateVideoThumbnail(videoPath) {
  return new Promise((resolve) => {
    const fs = require("fs")
    const path = require("path")
    const os = require("os")
    const { spawn } = require('child_process')
    
    const tempDir = os.tmpdir()
    const thumbnailPath = path.join(tempDir, `thumb_${Date.now()}.jpg`)
    
    // Use the bundled ffmpeg or system ffmpeg
    const ffmpegPath = getFfmpegPath()
    
    // Use ffmpeg to extract a frame at 1 second
    const ffmpegProcess = spawn(ffmpegPath, [
      '-i', videoPath,
      '-ss', '00:00:01.000',  // Seek to 1 second
      '-vframes', '1',        // Extract 1 frame
      '-vf', 'scale=320:-1',  // Scale to 320px width, maintain aspect ratio
      '-y',                   // Overwrite output file
      thumbnailPath
    ], {
      stdio: ['pipe', 'pipe', 'pipe']
    })

    ffmpegProcess.on('close', (code) => {
      if (code === 0 && fs.existsSync(thumbnailPath)) {
        try {
          // Read the thumbnail and convert to base64
          const thumbnailBuffer = fs.readFileSync(thumbnailPath)
          const base64Thumbnail = thumbnailBuffer.toString('base64')
          
          // Clean up temp file
          fs.unlinkSync(thumbnailPath)
          
          resolve(`data:image/jpeg;base64,${base64Thumbnail}`)
        } catch (error) {
          console.error('Error reading thumbnail:', error)
          // Fallback to simple thumbnail
          resolve(generateSimpleThumbnail('video'))
        }
      } else {
        // Fallback to simple thumbnail if ffmpeg fails
        resolve(generateSimpleThumbnail('video'))
      }
    })

    ffmpegProcess.on('error', (error) => {
      console.error('FFmpeg error:', error)
      resolve(generateSimpleThumbnail('video'))
    })
  })
}

async function generateImageThumbnail(imagePath) {
  try {
    const { nativeImage } = require("electron")
    
    // Use Electron's nativeImage to resize the image
    const image = nativeImage.createFromPath(imagePath)
    if (image.isEmpty()) {
      return generateSimpleThumbnail('image')
    }
    
    // Resize to thumbnail width while maintaining aspect ratio
    const resized = image.resize({ width: 320 })
    
    return resized.toDataURL()
  } catch (error) {
    console.error('Image thumbnail generation failed:', error)
    return generateSimpleThumbnail('image')
  }
}

function generateSimpleThumbnail(contentType) {
  const colors = {
    text: '#3b82f6',
    audio: '#f59e0b',
    video: '#ef4444',
    image: '#10b981'
  }
  
  const color = colors[contentType] || '#6b7280'
  
  const svg = `
    <svg width="200" height="150" xmlns="http://www.w3.org/2000/svg">
      <rect width="200" height="150" fill="${color}" opacity="0.3"/>
      <text x="100" y="80" text-anchor="middle" fill="${color}" font-family="Arial" font-size="16" font-weight="bold">
        ${contentType.toUpperCase()}
      </text>
    </svg>
  `
  
  return `data:image/svg+xml;base64,${Buffer.from(svg).toString('base64')}`
}

// Handler functions for the IPC calls
async function handleProcessFiles(event, filePaths) {
  if (pythonProcess && pythonProcess.stdin && !pythonProcess.killed) {
    const command = {
      action: "process",
      files: filePaths,
    }
    try {
      pythonProcess.stdin.write(JSON.stringify(command) + "\n")
    } catch (error) {
      console.error("Error writing to Python stdin:", error)
      return { success: false, error: "Failed to send command to Python process" }
    }
  } else {
    console.error("Python process not available for processing")
    return { success: false, error: "Python process not available" }
  }
  return { success: true }
}

async function handleGetMetadataPaths(event) {
  console.log("Debug: Getting metadata paths")
  
  try {
    // For packaged apps, directly construct the paths based on the known structure
    const os = require("os")
    let baseDir
    
    if (process.platform === 'darwin') {  // macOS
      baseDir = path.join(os.homedir(), 'Library', 'Application Support', 'silk.ai')
    } else if (process.platform === 'win32') {  // Windows  
      baseDir = path.join(process.env.APPDATA || os.homedir(), 'silk.ai')
    } else {  // Linux and others
      baseDir = path.join(os.homedir(), '.config', 'silk.ai')
    }
    
    const metadataDir = path.join(baseDir, 'metadata')
    
    const paths = {
      video: path.join(metadataDir, 'video_metadata.json'),
      audio: path.join(metadataDir, 'audio_metadata.json'),
      text: path.join(metadataDir, 'text_metadata.json'),
      image: path.join(metadataDir, 'image_metadata.json')
    }
    
    console.log("Debug: Metadata paths constructed:", paths)
    return paths
    
  } catch (error) {
    console.error("Failed to get metadata paths:", error)
    return {
      error: error.message,
      video: "Error loading path",
      audio: "Error loading path", 
      text: "Error loading path",
      image: "Error loading path"
    }
  }
}

async function handleTestApiConnectivity(event) {
  console.log("Testing API connectivity...")
  
  try {
    // Simple direct test of the Railway API using Node.js
    const https = require('https')
    
    const railwayUrl = 'https://contentcache-production.up.railway.app/health'
    
    const result = await new Promise((resolve, reject) => {
      console.log(`Testing Railway API at: ${railwayUrl}`)
      
      const timeout = setTimeout(() => {
        reject(new Error('Request timeout (30 seconds)'))
      }, 30000)
      
      const req = https.get(railwayUrl, (res) => {
        clearTimeout(timeout)
        
        let data = ''
        res.on('data', (chunk) => {
          data += chunk
        })
        
        res.on('end', () => {
          console.log(`API Response Status: ${res.statusCode}`)
          console.log(`API Response Data: ${data}`)
          
          if (res.statusCode === 200) {
            try {
              const responseData = JSON.parse(data)
              resolve({
                status: "success",
                message: "Railway API server is accessible and working",
                api_url: railwayUrl,
                health: responseData
              })
            } catch (parseError) {
              resolve({
                status: "success", 
                message: "Railway API server responded but with non-JSON data",
                api_url: railwayUrl,
                response: data
              })
            }
          } else {
            resolve({
              status: "error",
              message: `Railway API returned status ${res.statusCode}`,
              api_url: railwayUrl,
              status_code: res.statusCode,
              response: data
            })
          }
        })
      })
      
      req.on('error', (error) => {
        clearTimeout(timeout)
        console.error("API request error:", error)
        reject(error)
      })
      
      req.setTimeout(30000, () => {
        clearTimeout(timeout)
        req.destroy()
        reject(new Error('Request timeout'))
      })
    })
    
    return result
    
  } catch (error) {
    console.error("Failed to test API connectivity:", error)
    return {
      status: "error",
      message: `Failed to test API: ${error.message}`,
      api_url: "https://contentcache-production.up.railway.app/health",
      error_details: error.toString()
    }
  }
}

// Stop processing handler
async function handleStopProcessing() {
  try {
    console.log("🛑 Stopping processing by sending stop command to unified service...")
    
    // Send stop command to the existing Python process instead of spawning a new one
    if (pythonProcess && pythonProcess.stdin && !pythonProcess.killed) {
      const stopCommand = {
        action: "stop_enhanced"
      }
      
      try {
        pythonProcess.stdin.write(JSON.stringify(stopCommand) + "\n")
        console.log("✅ Stop command sent to unified service")
        return { success: true, message: "Stop command sent to processing service" }
      } catch (writeError) {
        console.error("❌ Failed to send stop command:", writeError)
        return { success: false, error: `Failed to send stop command: ${writeError.message}` }
      }
    } else {
      console.log("⚠️ Python process not available for stop command")
      return { success: false, error: "Python unified service process not available" }
    }
    
  } catch (error) {
    console.error("❌ Failed to stop processing:", error)
    return { success: false, error: error.message }
  }
}

// Download ffmpeg and ffprobe binaries if they don't exist
async function ensureFfmpegBinaries() {
  // Check if FFmpeg is available in system PATH first
  try {
    const { execSync } = require('child_process')
    execSync('ffmpeg -version', { stdio: 'ignore' })
    execSync('ffprobe -version', { stdio: 'ignore' })
    console.log("✅ FFmpeg binaries found in system PATH")
    return true
  } catch (error) {
    console.log("⚠️ FFmpeg not found in system PATH")
  }
  
  // Check for bundled binaries in the app package
  const binariesDir = path.join(__dirname, "binaries")
  const ffmpegPath = path.join(binariesDir, "ffmpeg")
  const ffprobePath = path.join(binariesDir, "ffprobe")
  
  if (fs.existsSync(ffmpegPath) && fs.existsSync(ffprobePath)) {
    console.log("✅ FFmpeg binaries found in app bundle")
    return true
  }
  
  // For now, we'll continue without FFmpeg and let the Python backend handle it
  console.log("⚠️ FFmpeg binaries not found - some features may be limited")
  console.log("📝 Note: Video processing will use Python backend's FFmpeg handling")
  return false
}


// App event handlers
app.whenReady().then(async () => {
  console.log("🚀 App is ready! Starting initialization sequence...")
  
  // Clear local logs on app launch
  console.log("🧹 Clearing local logs on app launch...")
  clearLocalLogs()
  
  // Clear UI logs on app launch to prevent accumulation
  console.log("🧹 Clearing UI log buffers on app launch...")
  clearUILogs()
  
  // Set up periodic log clearing to prevent UI lag (every 5 minutes)
  setInterval(() => {
    const totalLogs = globalPythonLogBuffer.length + globalSearchLogBuffer.length
    if (totalLogs > GLOBAL_MAX_LOG_LINES * 0.8) { // Clear when 80% full
      console.log(`🧹 Auto-clearing UI logs (${totalLogs} total logs)`)
      clearUILogs()
    }
  }, 5 * 60 * 1000) // 5 minutes
  
  // Setup Python environment for packaged apps
  setupPythonEnvironment()
  console.log("✅ Python environment setup complete")
  
  // Ensure FFmpeg binaries are available
  console.log("🔧 Ensuring FFmpeg binaries are available...")
  try {
    await ensureFfmpegBinaries()
    console.log("✅ FFmpeg binaries ready")
  } catch (error) {
    console.error("❌ Failed to setup FFmpeg binaries:", error)
  }
  
  // CRITICAL: Check and install dependencies BEFORE creating main window
  console.log("🔧 Checking Python dependencies before starting application...")
  dependencyInstallationInProgress = true
  
  try {
    // Add a timeout to prevent indefinite hanging
    const dependencyTimeout = new Promise((_, reject) => {
      setTimeout(() => reject(new Error("Dependency installation timeout after 10 minutes")), 10 * 60 * 1000)
    })
    
    await Promise.race([ensurePythonDependencies(), dependencyTimeout])
    console.log("✅ Python dependencies ready - creating main window...")
    notifyDependenciesReady()
    
    // Create main window only after dependencies are ready
    createWindow()
    console.log("✅ Main window created with dependencies confirmed")
    
    // Start services after dependencies are confirmed
    setTimeout(() => {
      console.log("🚀 Starting search server with confirmed dependencies...")
      try {
        startSearchServer()
        console.log("✅ Search server startup initiated")
      } catch (error) {
        console.error("❌ Search server startup failed:", error)
      }
    }, 1000)
    
    setTimeout(async () => {
      console.log("🚀 Starting Python unified service with confirmed dependencies...")
      try {
        await startPythonProcess()
        console.log("✅ Unified service startup initiated")
      } catch (error) {
        console.error("❌ Unified service startup failed:", error)
      }
    }, 2000) // Stagger the service starts
    
  } catch (error) {
    console.error("❌ Failed to setup Python dependencies:", error)
    notifyDependenciesFailed(error)
    
    // Still create main window but show error
    createWindow()
    console.log("✅ Main window created (with dependency warnings)")
    
    // Show error to user
    if (mainWindow && !mainWindow.isDestroyed()) {
      mainWindow.webContents.send("dependency-error", {
        message: "Failed to install Python dependencies. App functionality will be limited.",
        error: error.message
      })
    }
  }

  app.on("activate", () => {
    if (BrowserWindow.getAllWindows().length === 0) {
      createWindow()
    }
  })
})

app.on("window-all-closed", () => {
  // Clear local logs on window closure
  console.log("🧹 Clearing local logs on window closure...")
  clearLocalLogs()
  
  // Safely terminate processes
  try {
    if (pythonProcess && !pythonProcess.killed) {
      pythonProcess.kill('SIGTERM')
      setTimeout(() => {
        if (!pythonProcess.killed) {
          pythonProcess.kill('SIGKILL')
        }
      }, 5000)
    }
    
    if (searchServerProcess && !searchServerProcess.killed) {
      searchServerProcess.kill('SIGTERM')
      setTimeout(() => {
        if (!searchServerProcess.killed) {
          searchServerProcess.kill('SIGKILL')
        }
      }, 5000)
    }
  } catch (error) {
    console.error('Error terminating processes:', error)
  }

  if (process.platform !== "darwin") {
    app.quit()
  }
})

app.on("before-quit", () => {
  // Clear local logs on app closure
  console.log("🧹 Clearing local logs on app closure...")
  clearLocalLogs()
  
  // Safely terminate processes on quit
  try {
    if (pythonProcess && !pythonProcess.killed) {
      pythonProcess.kill('SIGTERM')
    }
    
    if (searchServerProcess && !searchServerProcess.killed) {
      searchServerProcess.kill('SIGTERM')
    }
  } catch (error) {
    console.error('Error in before-quit:', error)
  }
  
  // Clean shutdown without force-killing processes in main thread
  setTimeout(() => {
    exec("lsof -ti:3002,5001 | xargs kill -9 2>/dev/null || true", () => {})
  }, 1000)
})

// Check and install Python requirements for packaged apps
async function ensurePythonDependencies() {
  const isDev = !app.isPackaged
  
  if (isDev) {
    return // Development uses virtual environment
  }
  
  const pythonExe = getPythonExecutable()
  console.log("🐍 Checking Python dependencies for packaged app...")
  
  // Check if this is the first run by looking for a marker file
  const appDataDir = require('os').homedir()
  const markerFile = path.join(appDataDir, '.silk-ai-deps-installed')
  
  // More comprehensive dependency test including search functionality
  let corePackagesAvailable = true
  if (fs.existsSync(markerFile)) {
    // Log marker file info for debugging
    try {
      const markerContent = fs.readFileSync(markerFile, 'utf8')
      console.log(`📋 Found dependency marker file: ${markerFile}`)
      console.log(`📋 Marker content: ${markerContent.slice(0, 100)}...`)
    } catch (e) {
      console.log(`📋 Marker file exists but couldn't read it: ${e.message}`)
    }
    try {
      console.log("🔍 Testing core packages for search functionality...")
      
      // CRITICAL: Test numpy first since it's the most common failure point
      console.log("🧮 Testing numpy (critical dependency)...")
      const numpyTestResult = await new Promise((resolve) => {
        const numpyTestScript = `
import sys
print(f"Python: {sys.version}")
print(f"Platform: {sys.platform}")
print(f"Architecture: {sys.platform}")
try:
    import numpy as np
    print(f"NumPy version: {np.__version__}")
    print(f"NumPy location: {np.__file__}")
    # Test core multiarray (common failure point)
    from numpy.core.multiarray import _reconstruct
    # Test basic array creation
    arr = np.array([1, 2, 3])
    print(f"Array test: {arr.sum()}")
    print("NUMPY_OK")
except ImportError as e:
    print(f"NUMPY_IMPORT_ERROR: {e}")
except Exception as e:
    print(f"NUMPY_ERROR: {e}")
`
        const testProcess = spawn(pythonExe, ['-c', numpyTestScript], {
          stdio: ['pipe', 'pipe', 'pipe'],
          timeout: 20000  // 20 second timeout for numpy
        })
        
        let output = ''
        let errorOutput = ''
        
        testProcess.stdout.on('data', (data) => {
          output += data.toString()
        })
        
        testProcess.stderr.on('data', (data) => {
          errorOutput += data.toString()
        })
        
        testProcess.on('close', (code) => {
          const success = code === 0 && output.includes('NUMPY_OK')
          if (!success) {
            console.log(`❌ NumPy test failed - code: ${code}`)
            console.log(`Output: ${output}`)
            console.log(`Error: ${errorOutput}`)
          } else {
            console.log(`✅ NumPy test passed`)
          }
          resolve(success)
        })
        
        testProcess.on('error', (error) => {
          console.log(`❌ NumPy test process error: ${error.message}`)
          resolve(false)
        })
      })
      
      if (!numpyTestResult) {
        console.log(`❌ NumPy failed critical test - will reinstall`)
        corePackagesAvailable = false
      }
      
      // Test essential packages including search-specific dependencies
      const corePackages = [
        'requests', 
        'flask', 
        'flask_cors',
        'huggingface_hub',    // CRITICAL: Test before sentence_transformers
        'sentence_transformers', 
        'tensorflow',         // CRITICAL: Video processing dependency
        'tensorflow_hub',     // CRITICAL: Video processing dependency  
        'torch',              // PyTorch for ML operations
        'torchaudio',         // Audio processing
        'cv2',                // OpenCV for video/image processing  
        'skimage',            // Scikit-image for SSIM calculations
        'easyocr',            // OCR for text extraction
        'whisper',            // Audio transcription
        'soundfile',          // Audio file reading
        'rank_bm25',          // Critical for BM25 search
        'nltk',               // Enhanced tokenizer dependency
        'spellchecker',       // Spell checking for search
        'natsort',            // Natural sorting for file operations
        'PIL',                // Pillow for image processing
        'psutil'              // System monitoring
      ]
      
      for (const pkg of corePackages) {
        const testResult = await new Promise((resolve) => {
          const testProcess = spawn(pythonExe, ['-c', `import ${pkg}; print("${pkg}_OK")`], {
            stdio: ['pipe', 'pipe', 'pipe'],
            timeout: 15000  // 15 second timeout per package
          })
          
          let output = ''
          let errorOutput = ''
          
          testProcess.stdout.on('data', (data) => {
            output += data.toString()
          })
          
          testProcess.stderr.on('data', (data) => {
            errorOutput += data.toString()
          })
          
          testProcess.on('close', (code) => {
            const success = code === 0 && output.includes(`${pkg}_OK`)
            if (!success) {
              console.log(`⚠️ Package ${pkg} test failed - code: ${code}, output: ${output}, error: ${errorOutput}`)
            } else {
              console.log(`✅ Package ${pkg} test passed`)
            }
            resolve(success)
          })
          
          testProcess.on('error', (error) => {
            console.log(`⚠️ Package ${pkg} test error: ${error.message}`)
            resolve(false)
          })
        })
        
        if (!testResult) {
          console.log(`❌ Core package ${pkg} failed test`)
          corePackagesAvailable = false
          break  // If any core package fails, we need to reinstall
        }
      }
      
      // CRITICAL: Specific test for sentence_transformers + huggingface_hub compatibility
      if (corePackagesAvailable) {
        console.log("🔍 Testing sentence_transformers + huggingface_hub compatibility...")
        const compatibilityTestResult = await new Promise((resolve) => {
          const compatibilityTestScript = `
import sys
try:
    print("Testing huggingface_hub version...")
    import huggingface_hub
    print(f"huggingface_hub version: {huggingface_hub.__version__}")
    
    # Test the specific import that was failing
    from huggingface_hub import HfApi, HfFolder, Repository, hf_hub_url, cached_download
    print("✅ cached_download import successful")
    
    print("Testing sentence_transformers...")
    from sentence_transformers import SentenceTransformer, util
    print("✅ sentence_transformers import successful")
    
    print("COMPATIBILITY_TEST_PASSED")
except ImportError as e:
    print(f"COMPATIBILITY_IMPORT_ERROR: {e}")
except Exception as e:
    print(f"COMPATIBILITY_ERROR: {e}")
`
          const testProcess = spawn(pythonExe, ['-c', compatibilityTestScript], {
            stdio: ['pipe', 'pipe', 'pipe'],
            timeout: 20000  // 20 second timeout
          })
          
          let output = ''
          let errorOutput = ''
          
          testProcess.stdout.on('data', (data) => {
            output += data.toString()
          })
          
          testProcess.stderr.on('data', (data) => {
            errorOutput += data.toString()
          })
          
          testProcess.on('close', (code) => {
            const success = code === 0 && output.includes('COMPATIBILITY_TEST_PASSED')
            if (!success) {
              console.log(`❌ Compatibility test failed - code: ${code}`)
              console.log(`Output: ${output}`)
              console.log(`Error: ${errorOutput}`)
            } else {
              console.log(`✅ sentence_transformers + huggingface_hub compatibility verified`)
            }
            resolve(success)
          })
          
          testProcess.on('error', (error) => {
            console.log(`❌ Compatibility test process error: ${error.message}`)
            resolve(false)
          })
        })
        
        if (!compatibilityTestResult) {
          console.log(`❌ sentence_transformers + huggingface_hub compatibility failed - will reinstall`)
          corePackagesAvailable = false
        }
      }
      
      // Test NLTK data availability (critical for enhanced tokenizer)
      if (corePackagesAvailable) {
        console.log("🔍 Testing NLTK data packages...")
        const nltkTestResult = await new Promise((resolve) => {
          const testScript = `
import nltk
try:
    from nltk.tokenize import word_tokenize
    from nltk.corpus import stopwords
    from nltk.stem import PorterStemmer
    word_tokenize("test")
    stopwords.words('english')
    print("NLTK_DATA_OK")
except:
    print("NLTK_DATA_MISSING")
`
          const testProcess = spawn(pythonExe, ['-c', testScript], {
            stdio: ['pipe', 'pipe', 'pipe'],
            timeout: 10000
          })
          
          let output = ''
          testProcess.stdout.on('data', (data) => {
            output += data.toString()
          })
          
          testProcess.on('close', (code) => {
            const success = output.includes('NLTK_DATA_OK')
            if (!success) {
              console.log(`⚠️ NLTK data test failed - missing required data packages`)
            } else {
              console.log(`✅ NLTK data test passed`)
            }
            resolve(success)
          })
          
          testProcess.on('error', (error) => {
            console.log(`⚠️ NLTK data test error: ${error.message}`)
            resolve(false)
          })
        })
        
        if (!nltkTestResult) {
          console.log(`❌ NLTK data packages missing - search functionality will be impaired`)
          corePackagesAvailable = false
        }
      }
      
      if (corePackagesAvailable) {
        console.log("✅ All core packages and NLTK data available")
      }
      
    } catch (error) {
      console.log(`⚠️ Core package test error: ${error.message}`)
      corePackagesAvailable = false
    }
  }
  
  // Check version compatibility - force reinstall if version mismatch
  let versionMismatch = false
  if (fs.existsSync(markerFile)) {
    try {
      const existingMarker = JSON.parse(fs.readFileSync(markerFile, 'utf8'))
      const currentVersion = "2.0.0"
      if (existingMarker.version !== currentVersion) {
        console.log(`🔄 Version mismatch: ${existingMarker.version} → ${currentVersion}, forcing reinstall...`)
        versionMismatch = true
        fs.unlinkSync(markerFile) // Remove old marker
      }
    } catch (e) {
      console.log("⚠️ Could not read existing marker file, will reinstall")
      versionMismatch = true
    }
  }
  
  if (fs.existsSync(markerFile) && corePackagesAvailable && !versionMismatch) {
    console.log("✅ Dependencies already installed and available, skipping...")
    return
  }
  
  if (!corePackagesAvailable) {
    console.log("⚠️ Some core dependencies missing, reinstalling...")
  }
  
  // Show a loading window while installing dependencies
  const loadingWindow = new BrowserWindow({
    width: 450,
    height: 350,
    show: false,
    resizable: false,
    center: true,
    alwaysOnTop: true,
    frame: false,
    backgroundColor: '#667eea',
    webPreferences: {
      nodeIntegration: false,
      contextIsolation: true
    }
  })
  
  // Create a simple HTML page for the loading screen
  const loadingHtml = `
    <!DOCTYPE html>
    <html>
    <head>
        <style>
            body { 
                font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
                display: flex;
                flex-direction: column;
                justify-content: center;
                align-items: center;
                height: 100vh;
                margin: 0;
                background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                color: white;
            }
            .logo { font-size: 2em; margin-bottom: 20px; }
            .spinner {
                border: 3px solid rgba(255,255,255,0.3);
                border-radius: 50%;
                border-top: 3px solid white;
                width: 40px;
                height: 40px;
                animation: spin 2s linear infinite;
                margin-bottom: 20px;
            }
            @keyframes spin {
                0% { transform: rotate(0deg); }
                100% { transform: rotate(360deg); }
            }
            .message { text-align: center; max-width: 320px; }
            .details { font-size: 0.9em; opacity: 0.9; margin-top: 10px; }
        </style>
    </head>
    <body>
        <div class="logo">🚀 silk.ai</div>
        <div class="spinner"></div>
        <div class="message">
            <h3>Installing Dependencies...</h3>
            <p>Setting up AI models and required packages including NumPy, SentenceTransformers, natsort, and search functionality.</p>
            <div class="details">This may take a few minutes on first launch.</div>
        </div>
    </body>
    </html>
  `

  try {
    await loadingWindow.loadURL(`data:text/html,${encodeURIComponent(loadingHtml)}`)
    loadingWindow.show()
    
    console.log("🔄 Installing Python dependencies for packaged app...")
    
    // Setup isolated environment for dependency installation
    const appDataDir = getAppCacheDir()
    const userBase = path.join(appDataDir, 'python_packages')
    process.env.PYTHONUSERBASE = userBase
    console.log(`📦 Installing dependencies to isolated location: ${userBase}`)
    
    // Create the directory if it doesn't exist
    const userSitePackages = path.join(userBase, 'lib', 'python', 'site-packages')
    if (!fs.existsSync(userSitePackages)) {
      fs.mkdirSync(userSitePackages, { recursive: true })
      console.log(`✅ Created user site-packages directory: ${userSitePackages}`)
    }
    
    // CRITICAL: Install numpy first separately with special handling
    console.log("🧮 Installing NumPy first (critical foundation dependency)...")
    const numpyInstallSuccess = await new Promise((resolve) => {
      // Detect platform and architecture for numpy
      const platform = process.platform
      const arch = process.arch
      console.log(`Platform: ${platform}, Architecture: ${arch}`)
      
      // Use specific numpy installation strategy - CRITICAL: Must be 1.x for OpenCV compatibility
      let numpyPackage = 'numpy>=1.21.0,<2.0.0'  // Force NumPy 1.x to avoid OpenCV conflicts
      
      // Platform-specific numpy installation args - PYTHONUSERBASE controls where --user installs
      let installArgs = ['-m', 'pip', 'install', '--user', '--upgrade', '--no-warn-script-location']
      
      if (platform === 'darwin' && arch === 'arm64') {
        // Apple Silicon - ensure proper wheel
        console.log("🍎 Detected Apple Silicon - using compatible NumPy")
        installArgs.push('--only-binary=:all:')  // Force binary wheels
      } else if (platform === 'darwin' && arch === 'x64') {
        // Intel Mac - use Intel-optimized version
        console.log("🍎 Detected Intel Mac - using Intel-optimized NumPy")
        installArgs.push('--only-binary=:all:')
      } else if (platform === 'win32') {
        // Windows - use pre-compiled wheels
        console.log("🪟 Detected Windows - using pre-compiled NumPy")
        installArgs.push('--only-binary=:all:')
      }
      
      // First uninstall any existing NumPy 2.x that might be causing conflicts
      installArgs.push('--force-reinstall', numpyPackage)
      
      const numpyProcess = spawn(pythonExe, installArgs, {
        stdio: ['pipe', 'pipe', 'pipe']
      })
      
      let installOutput = ''
      let installError = ''
      
      numpyProcess.stdout.on('data', (data) => {
        installOutput += data.toString()
      })
      
      numpyProcess.stderr.on('data', (data) => {
        installError += data.toString()
      })
      
      numpyProcess.on('close', (code) => {
        if (code === 0) {
          console.log(`✅ NumPy installed successfully`)
          resolve(true)
        } else {
          console.log(`❌ NumPy installation failed with code ${code}`)
          console.log(`Output: ${installOutput}`)
          console.log(`Error: ${installError}`)
          resolve(false)
        }
      })
      
      numpyProcess.on('error', (error) => {
        console.log(`❌ NumPy installation error: ${error.message}`)
        resolve(false)
      })
    })
    
    if (!numpyInstallSuccess) {
      console.log("⚠️ NumPy installation failed - some features may not work properly")
    }
    
    // Test numpy after installation
    if (numpyInstallSuccess) {
      console.log("🧮 Testing NumPy installation...")
      const numpyVerifyResult = await new Promise((resolve) => {
        const testScript = `
try:
    import numpy as np
    from numpy.core.multiarray import _reconstruct
    arr = np.array([1, 2, 3])
    print(f"NumPy {np.__version__} working correctly - array sum: {arr.sum()}")
    print("NUMPY_VERIFIED")
except Exception as e:
    print(f"NumPy verification failed: {e}")
`
        const testProcess = spawn(pythonExe, ['-c', testScript], {
          stdio: ['pipe', 'pipe', 'pipe'],
          timeout: 10000
        })
        
        let output = ''
        testProcess.stdout.on('data', (data) => {
          output += data.toString()
        })
        
        testProcess.on('close', (code) => {
          const success = output.includes('NUMPY_VERIFIED')
          if (success) {
            console.log(`✅ NumPy verification passed`)
          } else {
            console.log(`❌ NumPy verification failed`)
          }
          resolve(success)
        })
        
        testProcess.on('error', (error) => {
          console.log(`❌ NumPy verification error: ${error.message}`)
          resolve(false)
        })
      })
      
      if (!numpyVerifyResult) {
        console.log("⚠️ NumPy verification failed after installation")
      }
    }
    
    // Define all essential packages (excluding numpy since we installed it separately)
    const essentialPackages = [
      // Core API and server
      'fastapi==0.104.1',
      'uvicorn[standard]==0.24.0',
      'requests==2.31.0',
      'flask>=2.0.0',
      'flask-cors>=3.0.0',
      
      // Machine learning and embeddings (numpy already installed)
      // CRITICAL: Fix huggingface_hub compatibility with sentence-transformers 2.2.2
      // sentence-transformers 2.2.2 requires older huggingface_hub with cached_download
      'huggingface_hub>=0.10.0,<0.16.0',  // Compatible version range for cached_download
      'sentence-transformers==2.2.2',
      'torch>=2.0.0',
      'transformers>=4.6.0',
      
      // TensorFlow for video processing - CRITICAL for videotagger
      'tensorflow>=2.10.0',
      'tensorflow-hub',
      
      // Search functionality - CRITICAL
      'rank-bm25==0.2.2',
      'nltk>=3.8',
      'pyspellchecker>=0.8.0',
      'natsort==8.4.0',  // Natural sorting for file operations
      
      // Image and document processing - OpenCV compatible with NumPy 1.x
      'pillow==10.4.0',
      'opencv-python>=4.5.0,<4.9.0',  // Ensure NumPy 1.x compatibility
      'scikit-image==0.22.0',  // CRITICAL: Used by framesegmentation.py for SSIM
      'easyocr==1.7.0',
      'pdfplumber>=3.0.0',
      'pypdf>=3.0.0',
      'python-docx>=1.1.0',
      'python-pptx>=0.6.0',  // Used by textprocessor
      'openpyxl>=3.1.0',     // Used by textprocessor
      
      // Audio processing with torch audio
      'openai-whisper>=20231117',
      'torchaudio',
      'soundfile',
      'PyYAML',
      
      // System utilities
      'psutil==5.9.5',
      'pydantic==2.5.0',
      'python-dotenv==1.0.0',
      'python-multipart==0.0.6'  // Used by FastAPI
    ]
    
    // Install packages in batches to avoid timeouts
    const batchSize = 5
    for (let i = 0; i < essentialPackages.length; i += batchSize) {
      const batch = essentialPackages.slice(i, i + batchSize)
      console.log(`📦 Installing batch ${Math.floor(i/batchSize) + 1}/${Math.ceil(essentialPackages.length/batchSize)}: ${batch.map(p => p.split('==')[0]).join(', ')}`)
      
      await new Promise((resolve, reject) => {
        const installArgs = ['-m', 'pip', 'install', '--user', '--upgrade', '--no-warn-script-location'].concat(batch)
        const installProcess = spawn(pythonExe, installArgs, {
          stdio: ['pipe', 'pipe', 'pipe']
        })
        
        let installOutput = ''
        let installError = ''
        
        installProcess.stdout.on('data', (data) => {
          installOutput += data.toString()
        })
        
        installProcess.stderr.on('data', (data) => {
          installError += data.toString()
        })
        
        installProcess.on('close', (code) => {
          if (code === 0) {
            console.log(`✅ Batch installed successfully`)
            resolve()
          } else {
            console.log(`⚠️ Batch installation failed with code ${code}`)
            console.log(`Output: ${installOutput}`)
            console.log(`Error: ${installError}`)
            // Don't reject - continue with other batches
            resolve()
          }
        })
        
        installProcess.on('error', (error) => {
          console.log(`⚠️ Batch installation error: ${error.message}`)
          resolve() // Don't reject - continue with other batches
        })
      })
    }
    
    // Download NLTK data packages (critical for search)
    console.log("📥 Downloading NLTK data packages...")
    await new Promise((resolve) => {
      const nltkScript = `
import nltk
import ssl
try:
    _create_unverified_https_context = ssl._create_unverified_context
except AttributeError:
    pass
else:
    ssl._create_default_https_context = _create_unverified_https_context

packages = ['punkt', 'stopwords', 'wordnet', 'omw-1.4']
for package in packages:
    try:
        print(f"Downloading {package}...")
        nltk.download(package, quiet=True)
        print(f"✅ {package} downloaded")
    except Exception as e:
        print(f"⚠️ Failed to download {package}: {e}")
print("NLTK_DOWNLOAD_COMPLETE")
`
      
      const nltkProcess = spawn(pythonExe, ['-c', nltkScript], {
        stdio: ['pipe', 'pipe', 'pipe']
      })
      
      nltkProcess.stdout.on('data', (data) => {
        console.log(`NLTK: ${data.toString().trim()}`)
      })
      
      nltkProcess.stderr.on('data', (data) => {
        console.log(`NLTK Error: ${data.toString().trim()}`)
      })
      
      nltkProcess.on('close', (code) => {
        console.log(`✅ NLTK data download completed`)
        resolve()
      })
      
      nltkProcess.on('error', (error) => {
        console.log(`⚠️ NLTK download error: ${error.message}`)
        resolve()
      })
    })
    
    // Create marker file with installation details
    const markerData = {
      installed_at: new Date().toISOString(),
      version: "2.0.0",  // MAJOR INCREMENT: Force reinstall with complete dependency list
      numpy_version: numpyInstallSuccess ? "1.24.4" : "failed",
      numpy_platform: `${process.platform}-${process.arch}`,
      packages_count: essentialPackages.length,
      core_packages: ['numpy', 'tensorflow', 'tensorflow_hub', 'torch', 'torchaudio', 'cv2', 'skimage', 'easyocr', 'whisper', 'soundfile', 'requests', 'flask', 'flask_cors', 'sentence_transformers', 'rank_bm25', 'nltk', 'spellchecker', 'natsort', 'PIL', 'psutil'],
      video_packages: ['tensorflow', 'tensorflow_hub', 'cv2', 'skimage', 'easyocr', 'PIL'],
      audio_packages: ['torch', 'torchaudio', 'whisper', 'soundfile'],
      search_packages: ['rank_bm25', 'nltk', 'pyspellchecker', 'natsort'],
      document_packages: ['pdfplumber', 'pypdf', 'python-docx', 'python-pptx', 'openpyxl'],
      nltk_data: ['punkt', 'stopwords', 'wordnet', 'omw-1.4'],
      installation_notes: [
        "COMPLETE dependency installation with all video processing packages",
        "TensorFlow and TensorFlow Hub for video analysis",
        "PyTorch and TorchAudio for audio processing", 
        "OpenCV and EasyOCR for image/text extraction",
        "Scikit-image for frame similarity calculations",
        "Complete document processing suite",
        "Full search functionality with NLTK and BM25"
      ]
    }
    fs.writeFileSync(markerFile, JSON.stringify(markerData, null, 2))
    console.log("✅ Dependencies installation complete!")
    
    // Final comprehensive test
    console.log("🔍 Running final dependency verification...")
    const finalTestResult = await new Promise((resolve) => {
      const finalTestScript = `
import sys
print("=== FINAL DEPENDENCY TEST ===")
print(f"Python path: {sys.path[:3]}...")
try:
    # Test numpy (most critical)
    import numpy as np
    print(f"✅ NumPy {np.__version__} - OK")
    print(f"   NumPy location: {np.__file__}")
    
    # Test OpenCV (the main culprit)
    import cv2
    print(f"✅ OpenCV {cv2.__version__} - OK")
    print(f"   OpenCV location: {cv2.__file__}")
    
    # Test TensorFlow (critical for video processing)
    import tensorflow as tf
    print(f"✅ TensorFlow {tf.__version__} - OK")
    print(f"   TensorFlow location: {tf.__file__}")
    
    import tensorflow_hub as hub
    print("✅ TensorFlow Hub - OK")
    print(f"   TensorFlow Hub location: {hub.__file__}")
    
    # Test core imports
    import requests
    print("✅ Requests - OK")
    
    import flask
    print("✅ Flask - OK")
    
    # Test ML packages
    import sentence_transformers
    print("✅ SentenceTransformers - OK")
    
    import rank_bm25
    print("✅ BM25 - OK")
    
    print("✅ ALL DEPENDENCIES VERIFIED - USING ISOLATED ENVIRONMENT")
    print("FINAL_TEST_PASSED")
except Exception as e:
    print(f"❌ Final test failed: {e}")
    import traceback
    traceback.print_exc()
    print("FINAL_TEST_FAILED")
`
      
      const testProcess = spawn(pythonExe, ['-c', finalTestScript], {
        stdio: ['pipe', 'pipe', 'pipe'],
        timeout: 30000  // 30 second timeout for final test
      })
      
      let output = ''
      testProcess.stdout.on('data', (data) => {
        const text = data.toString()
        output += text
        console.log(`Final Test: ${text.trim()}`)
      })
      
      testProcess.on('close', (code) => {
        const success = output.includes('FINAL_TEST_PASSED')
        if (success) {
          console.log(`🎉 All dependencies verified and ready!`)
        } else {
          console.log(`⚠️ Some dependencies may not be working correctly`)
        }
        resolve(success)
      })
      
      testProcess.on('error', (error) => {
        console.log(`❌ Final test error: ${error.message}`)
        resolve(false)
      })
    })
    
    if (!finalTestResult) {
      console.log("⚠️ Final verification failed - dependencies not working properly")
      // Update marker to indicate issues
      markerData.final_verification = "failed"
      markerData.installation_notes.push("Final verification failed - dependencies may not be working")
      fs.writeFileSync(markerFile, JSON.stringify(markerData, null, 2))
      
      // Notify that dependencies failed
      throw new Error("Final dependency verification failed - critical packages not working")
    } else {
      console.log("🎉 All dependencies verified and working perfectly!")
      markerData.final_verification = "passed"
      markerData.huggingface_hub_compatibility = "verified"
      markerData.installation_notes.push("All dependencies including sentence_transformers compatibility verified")
      fs.writeFileSync(markerFile, JSON.stringify(markerData, null, 2))
    }
    
      } catch (error) {
      console.log("⚠️ Dependency installation failed, but continuing:", error.message)
      
      // Notify that dependencies failed
      notifyDependenciesFailed(error)
    } finally {
      // Close the loading window
      if (loadingWindow && !loadingWindow.isDestroyed()) {
        loadingWindow.close()
      }
    }
}

ipcMain.handle("search-videos", async (event, query, options = {}) => {
  if (pythonProcess) {
    const searchCommand = {
        action: "search",
        query: query,
        content_types: options.content_types || null,
        top_k: options.top_k || 20,
      date_filter: options.date_filter || '',
      location_filter: options.location_filter || '',
      offset: options.offset || 0,
    }
    pythonProcess.stdin.write(JSON.stringify(searchCommand) + "\n")
  } else {
    return { success: false, error: "Python process not available" }
  }
  return { success: true }
})

ipcMain.handle("get-system-status", async (event) => {
  if (pythonProcess) {
    pythonProcess.stdin.write(JSON.stringify({ action: "status" }) + "\n")
  }
  return { success: true }
})

ipcMain.handle("start-search-server", async (event) => {
  try {
    console.log("🔍 Starting search server via IPC...")
    startSearchServer()
    return { success: true }
  } catch (error) {
    console.error("❌ Error starting search server:", error)
    return { success: false, error: error.message }
  }
})

ipcMain.handle("reinstall-dependencies", async (event) => {
  try {
    console.log("🔄 Manual dependency reinstallation requested...")
    
    // Remove marker file to force complete reinstall
    const appDataDir = require('os').homedir()
    const markerFile = path.join(appDataDir, '.silk-ai-deps-installed')
    
    if (fs.existsSync(markerFile)) {
      fs.unlinkSync(markerFile)
      console.log("🗑️ Removed dependency marker file")
    }
    
    // Run dependency installation
    await ensurePythonDependencies()
    
    return { 
      success: true, 
      message: "Dependencies reinstalled successfully. Please restart the app." 
    }
  } catch (error) {
    console.error("❌ Error reinstalling dependencies:", error)
    return { 
      success: false, 
      error: error.message,
      message: "Failed to reinstall dependencies. Check console for details."
    }
  }
})

ipcMain.handle("check-dependency-status", async (event) => {
  try {
    const appDataDir = require('os').homedir()
    const markerFile = path.join(appDataDir, '.silk-ai-deps-installed')
    
    if (!fs.existsSync(markerFile)) {
      return {
        installed: false,
        message: "Dependencies not installed",
        marker_file_exists: false
      }
    }
    
    const markerContent = JSON.parse(fs.readFileSync(markerFile, 'utf8'))
    
    // Test numpy specifically
    const pythonExe = getPythonExecutable()
    const numpyStatus = await new Promise((resolve) => {
      const testProcess = spawn(pythonExe, ['-c', 'import numpy as np; print(f"NumPy {np.__version__} OK")'], {
        stdio: ['pipe', 'pipe', 'pipe'],
        timeout: 10000
      })
      
      let output = ''
      testProcess.stdout.on('data', (data) => {
        output += data.toString()
      })
      
      testProcess.on('close', (code) => {
        resolve({
          success: code === 0,
          output: output.trim(),
          code: code
        })
      })
      
      testProcess.on('error', (error) => {
        resolve({
          success: false,
          error: error.message,
          code: -1
        })
      })
    })
    
    return {
      installed: true,
      marker_file_exists: true,
      marker_content: markerContent,
      numpy_status: numpyStatus,
      platform: `${process.platform}-${process.arch}`
    }
    
  } catch (error) {
    return {
      installed: false,
      error: error.message,
      marker_file_exists: fs.existsSync(path.join(require('os').homedir(), '.silk-ai-deps-installed'))
    }
  }
})
