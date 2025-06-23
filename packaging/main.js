const { app, BrowserWindow, dialog, ipcMain, Menu, shell } = require("electron")
const path = require("path")
const { spawn, exec } = require("child_process")
const fs = require("fs")

let mainWindow
let pythonProcess
let searchServerProcess
let isFirstLaunch = false

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
    // In packaged mode, ensure Python can find our modules
    const pythonPath = path.join(process.resourcesPath, "backend", "processing")
    process.env.PYTHONPATH = pythonPath + (process.env.PYTHONPATH ? `:${process.env.PYTHONPATH}` : '')
    
    // Also add the python scripts directory
    const pythonScriptsPath = path.join(process.resourcesPath, "python")
    process.env.PYTHONPATH += `:${pythonScriptsPath}`
    
    console.log(`Set PYTHONPATH for packaged app: ${process.env.PYTHONPATH}`)
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
    const modelDownloaderPath = path.join(__dirname, "python", "model_downloader.py")
    const pythonCmd = process.platform === 'win32' ? 'python' : 'python3'
    
    const downloadProcess = spawn(pythonCmd, [modelDownloaderPath, '--required-only'], {
      stdio: ['pipe', 'pipe', 'pipe'],
      cwd: __dirname
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
function startSearchServer() {
  try {
    console.log("🔍 DEBUG: startSearchServer() called")
    
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
      return
    }
    console.log(`✅ Search server script exists at: ${searchServerScript}`)
    
    // Check if the working directory exists
    if (!fs.existsSync(searchWorkingDir)) {
      console.error(`❌ Search working directory not found at: ${searchWorkingDir}`)
      return
    }
    console.log(`✅ Search working directory exists at: ${searchWorkingDir}`)
    
    console.log(`🚀 Starting search server with: ${pythonExe} ${searchServerScript}`)
    console.log(`🚀 Working directory: ${searchWorkingDir}`)
    
    searchServerProcess = spawn(pythonExe, [searchServerScript], {
      stdio: ["pipe", "pipe", "pipe"],
      cwd: searchWorkingDir
    })

    console.log(`✅ Search server process spawned with PID: ${searchServerProcess.pid}`)

    searchServerProcess.stdout.on("data", (data) => {
      const output = data.toString()
      console.log(`🔍 Search Server STDOUT: ${output}`)
      if (mainWindow && !mainWindow.isDestroyed()) {
      mainWindow.webContents.send("search-server-output", output)
      }
    })

    searchServerProcess.stderr.on("data", (data) => {
      const error = data.toString()
      console.error(`🔍 Search Server STDERR: ${error}`)
      if (mainWindow && !mainWindow.isDestroyed()) {
        mainWindow.webContents.send("search-server-error", error)
      }
    })

    searchServerProcess.on("close", (code) => {
      console.log(`🔍 Search server process closed with code ${code}`)
    })

    searchServerProcess.on("error", (error) => {
      console.error(`🔍 Search server process error:`, error)
    })

    console.log("✅ Search server startup completed (process handlers attached)")
  } catch (error) {
    console.error("❌ Failed to start search server:", error)
  }
}

// Python process management
function startPythonProcess() {
  try {
    const pythonExe = getPythonExecutable()
    const pythonScriptsPath = getPythonScriptsPath()
    const mainScript = path.join(pythonScriptsPath, "main.py")
    
    console.log(`Starting Python process with: ${pythonExe} ${mainScript}`)
    
    pythonProcess = spawn(pythonExe, [mainScript], {
      stdio: ["pipe", "pipe", "pipe"],
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

    pythonProcess.stdout.on("data", (data) => {
      const output = data.toString()
      pythonOutputBuffer += output
      
      // Process complete lines from the buffer
      const lines = pythonOutputBuffer.split('\n')
      pythonOutputBuffer = lines.pop() // Keep the incomplete line in buffer
      
      for (const line of lines) {
        if (line.trim() && mainWindow && !mainWindow.isDestroyed()) {
          mainWindow.webContents.send("python-output", line.trim())
        }
      }
    })

    pythonProcess.stderr.on("data", (data) => {
      console.error(`Python stderr: ${data}`)
      if (mainWindow && !mainWindow.isDestroyed()) {
      mainWindow.webContents.send("python-error", data.toString())
      }
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

// IPC handlers
ipcMain.handle("select-files", handleFileImport)
ipcMain.handle("select-folder", handleFolderImport)
ipcMain.handle("process-files", handleProcessFiles)
ipcMain.handle("get-metadata-paths", handleGetMetadataPaths)
ipcMain.handle("test-api-connectivity", handleTestApiConnectivity)
ipcMain.handle("stop-processing", handleStopProcessing)

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
    if (pythonProcess && !pythonProcess.killed) {
      console.log("🛑 Sending enhanced stop command to Python process...")
      
      // Send enhanced stop command
      pythonProcess.stdin.write(JSON.stringify({
        action: 'stop_enhanced'
      }) + '\n')
      
      console.log("✅ Enhanced stop command sent")
      return { success: true }
    } else {
      console.log("⚠️ No Python process running")
      return { success: false, message: "No Python process running" }
    }
  } catch (error) {
    console.error("❌ Failed to stop processing:", error)
    return { success: false, error: error.message }
  }
}

// App event handlers
app.whenReady().then(() => {
  console.log("🚀 App is ready! Starting initialization sequence...")
  
  // Setup Python environment for packaged apps
  setupPythonEnvironment()
  console.log("✅ Python environment setup complete")
  
  createWindow()
  console.log("✅ Main window created")
  
  // Start both services immediately, don't wait for dependencies
  console.log("🚀 Starting search server immediately...")
  setTimeout(() => {
    console.log("⏰ Search server startup timeout triggered - starting now")
    try {
      startSearchServer()
      console.log("✅ Search server startup initiated")
    } catch (error) {
      console.error("❌ Search server startup failed:", error)
    }
  }, 1000)
  
  console.log("🚀 Starting unified service...")
  setTimeout(() => {
    console.log("⏰ Unified service startup timeout triggered - starting now")
    try {
      startPythonProcess()
      console.log("✅ Unified service startup initiated")
    } catch (error) {
      console.error("❌ Unified service startup failed:", error)
    }
  }, 3000)
  
  // Handle dependencies in background without blocking service startup
  console.log("🔧 Starting Python dependencies check in background...")
  setTimeout(() => {
    ensurePythonDependencies()
      .then(() => {
        console.log("✅ Python dependencies ready")
      })
      .catch((error) => {
        console.error("❌ Failed to setup Python dependencies:", error)
        // Show error to user and continue with limited functionality
        if (mainWindow && !mainWindow.isDestroyed()) {
          mainWindow.webContents.send("dependency-error", {
            message: "Failed to install Python dependencies. Some features may not work.",
            error: error.message
          })
        }
      })
  }, 5000) // Start dependency check after services are running

  app.on("activate", () => {
    if (BrowserWindow.getAllWindows().length === 0) {
      createWindow()
    }
  })
})

app.on("window-all-closed", () => {
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
  
  // Also check if required packages are actually available
  let allPackagesAvailable = true
  if (fs.existsSync(markerFile)) {
    try {
      // Test if key packages are available - especially whisper which was failing
      const testResult = await new Promise((resolve) => {
        const testProcess = spawn(pythonExe, ['-c', 'import whisper, sentence_transformers, rank_bm25, flask, torch, transformers; print("OK")'], {
          stdio: ['pipe', 'pipe', 'pipe']
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
          if (code !== 0 || !output.includes('OK')) {
            console.log(`⚠️ Dependency test failed - code: ${code}, output: ${output}, error: ${errorOutput}`)
          }
          resolve(code === 0 && output.includes('OK'))
        })
        
        testProcess.on('error', (error) => {
          console.log(`⚠️ Dependency test error: ${error.message}`)
          resolve(false)
        })
      })
      
      allPackagesAvailable = testResult
    } catch (error) {
      allPackagesAvailable = false
    }
  }
  
  if (fs.existsSync(markerFile) && allPackagesAvailable) {
    console.log("✅ Dependencies already installed and available, skipping...")
    return
  }
  
  if (!allPackagesAvailable) {
    console.log("⚠️ Some dependencies missing, reinstalling...")
  }
  
  // Show a loading window while installing dependencies
  const loadingWindow = new BrowserWindow({
    width: 400,
    height: 300,
    show: false,
    resizable: false,
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
            .message { text-align: center; max-width: 300px; }
        </style>
    </head>
    <body>
        <div class="logo">🚀 silk.ai</div>
        <div class="spinner"></div>
        <div class="message">
            <h3>Setting up for first run...</h3>
            <p>Installing AI dependencies. This may take a few minutes.</p>
        </div>
    </body>
    </html>
  `
  
  loadingWindow.loadURL(`data:text/html;charset=utf-8,${encodeURIComponent(loadingHtml)}`)
  loadingWindow.show()
  
  // Essential packages needed for the app to run
  const essentialPackages = [
    'sentence-transformers',
    'torch',
    'torchaudio',
    'numpy',
    'requests',
    'flask',
    'flask-cors',
    'rank_bm25',           // Needed for search functionality
    'Pillow',              // Image processing
    'python-dotenv',       // Environment variables
    'pydantic',            // Data validation
    'scikit-learn',        // Machine learning utilities
    'opencv-python',       // Computer vision for frame processing
    'scikit-image',        // Image processing (for SSIM comparison)
    'natsort',             // Natural sorting for frame ordering
    'easyocr',             // OCR functionality for text extraction
    'psutil',              // System monitoring and memory usage
    'moondream',           // Image captioning
    'transformers',        // Hugging Face transformers
    'openai-whisper',      // CORRECTED: OpenAI Whisper for audio transcription
    'tensorflow',          // TensorFlow for ML models
    'tensorflow-hub',      // TensorFlow Hub for pre-trained models
    'soundfile',           // Audio file I/O
    'PyYAML',              // YAML parsing
  ]
  
  try {
    console.log("📦 Installing essential Python packages...")
    
    // First, explicitly uninstall conflicting packages to prevent issues
    console.log("🧹 Removing conflicting packages...")
    await new Promise((resolve) => {
      const uninstallProcess = spawn(pythonExe, ['-m', 'pip', 'uninstall', '--user', 'whisper', '-y'], {
        stdio: ['pipe', 'pipe', 'pipe']
      })
      
      uninstallProcess.on("close", (code) => {
        console.log(`✅ Removed conflicting whisper package (exit code: ${code})`)
        resolve() // Continue regardless of success
      })
      
      uninstallProcess.on("error", (error) => {
        console.log(`⚠️ Error removing whisper package (continuing):`, error.message)
        resolve() // Continue anyway
      })
    })
    
    // Install essential packages one by one with progress
    // Use --user flag to install to user directory that packaged apps can access
    for (let i = 0; i < essentialPackages.length; i++) {
      const pkg = essentialPackages[i]
      console.log(`Installing ${pkg} (${i + 1}/${essentialPackages.length})...`)
      
      await new Promise((resolve, reject) => {
        const installProcess = spawn(pythonExe, ['-m', 'pip', 'install', '--user', pkg], {
          stdio: ['pipe', 'pipe', 'pipe']
        })
        
        let output = ''
        let errorOutput = ''
        
        installProcess.stdout.on('data', (data) => {
          output += data.toString()
        })
        
        installProcess.stderr.on('data', (data) => {
          errorOutput += data.toString()
        })
        
        installProcess.on("close", (code) => {
          if (code === 0) {
            console.log(`✅ Installed ${pkg}`)
          } else {
            console.log(`⚠️ Failed to install ${pkg}:`, errorOutput)
          }
          resolve() // Continue even if one package fails
        })
        
        installProcess.on("error", (error) => {
          console.log(`⚠️ Error installing ${pkg}:`, error.message)
          resolve() // Continue anyway
        })
      })
    }
    
    // Create marker file to indicate dependencies are installed
    fs.writeFileSync(markerFile, new Date().toISOString())
    console.log("✅ Dependencies installation complete!")
    
  } catch (error) {
    console.log("⚠️ Dependency installation failed, but continuing:", error.message)
  } finally {
    // Close the loading window
    loadingWindow.close()
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
