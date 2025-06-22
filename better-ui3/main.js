const { app, BrowserWindow, dialog, ipcMain, Menu } = require("electron")
const path = require("path")
const { spawn, exec } = require("child_process")

let mainWindow
let pythonProcess
let searchServerProcess

// Function to kill processes on specific ports
function killProcessesOnPorts() {
  return new Promise((resolve) => {
    // Kill processes on ports 3002 (Next.js) and 5001 (search server)
    exec("lsof -ti:3002,5001 | xargs kill -9 2>/dev/null || true", (error) => {
      if (error) {
        console.log("No processes to kill or error killing processes:", error.message)
      } else {
        console.log("Killed existing processes on ports 3002 and 5001")
      }
      // Give processes time to clean up
      setTimeout(resolve, 2000)
    })
  })
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
    mainWindow.loadFile("out/index.html").catch((error) => {
      console.error("Failed to load production file:", error)
    })
  }

  // Show window when ready
  mainWindow.once("ready-to-show", () => {
    if (mainWindow && !mainWindow.isDestroyed()) {
    mainWindow.show()
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
      label: "VideoSearch Pro",
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
      { name: "Video Files", extensions: ["mp4", "mov", "avi", "mkv", "wmv", "flv", "webm"] },
      { name: "Audio Files", extensions: ["mp3", "wav", "aac", "flac", "m4a"] },
      { name: "All Files", extensions: ["*"] },
    ],
  })

  if (!result.canceled) {
    mainWindow.webContents.send("files-selected", result.filePaths)
  }
}

async function handleFolderImport() {
  const result = await dialog.showOpenDialog(mainWindow, {
    properties: ["openDirectory"],
  })

  if (!result.canceled) {
    mainWindow.webContents.send("folder-selected", result.filePaths[0])
  }
}

// Search server management
function startSearchServer() {
  try {
    // Start the search server from the backend directory using the virtual environment
    const backendPath = path.join(__dirname, "..", "backend", "search")
    const venvPython = path.join(__dirname, "..", ".venv", "bin", "python")
    searchServerProcess = spawn(venvPython, [path.join(backendPath, "search_server.py")], {
      stdio: ["pipe", "pipe", "pipe"],
      cwd: backendPath
    })

    searchServerProcess.stdout.on("data", (data) => {
      const output = data.toString()
      console.log(`Search Server: ${output}`)
      mainWindow.webContents.send("search-server-output", output)
    })

    searchServerProcess.stderr.on("data", (data) => {
      console.error(`Search Server Error: ${data}`)
      mainWindow.webContents.send("search-server-error", data.toString())
    })

    searchServerProcess.on("close", (code) => {
      console.log(`Search server exited with code ${code}`)
    })

    console.log("Search server started successfully")
  } catch (error) {
    console.error("Failed to start search server:", error)
  }
}

// Python process management
function startPythonProcess() {
  try {
    // Use the unified service main.py with virtual environment
    const venvPython = path.join(__dirname, "..", ".venv", "bin", "python")
    pythonProcess = spawn(venvPython, [path.join(__dirname, "python", "main.py")], {
      stdio: ["pipe", "pipe", "pipe"],
    })

    let pythonOutputBuffer = ''

    pythonProcess.stdout.on("data", (data) => {
      const output = data.toString()
      pythonOutputBuffer += output
      
      // Process complete lines from the buffer
      const lines = pythonOutputBuffer.split('\n')
      pythonOutputBuffer = lines.pop() // Keep the incomplete line in buffer
      
      for (const line of lines) {
        if (line.trim()) {
          mainWindow.webContents.send("python-output", line.trim())
        }
      }
    })

    pythonProcess.stderr.on("data", (data) => {
      console.error(`Python stderr: ${data}`)
      mainWindow.webContents.send("python-error", data.toString())
    })

    pythonProcess.on("close", (code) => {
      console.log(`Python process exited with code ${code}`)
    })

    // Send initial status request
    setTimeout(() => {
      if (pythonProcess) {
        pythonProcess.stdin.write(JSON.stringify({ action: "status" }) + "\n")
      }
    }, 1000)
  } catch (error) {
    console.error("Failed to start Python process:", error)
  }
}

// Enhanced IPC handlers
ipcMain.handle("process-files", async (event, filePaths) => {
  if (pythonProcess) {
    pythonProcess.stdin.write(
      JSON.stringify({
        action: "process",
        files: filePaths,
      }) + "\n",
    )
  }
  return { success: true }
})

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

// Simple thumbnail generation (like old UI)
ipcMain.handle("generate-thumbnail", async (event, filePath, contentType) => {
  try {
    // Generate thumbnail using the same approach as old UI
    const result = await generateSafeThumbnail(filePath, contentType)
    return result
  } catch (error) {
    console.error('Thumbnail generation failed:', error)
    return generateSimpleThumbnail(contentType)
  }
})

// No caching needed - generate thumbnails on demand like old UI

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
    
    // Use ffmpeg to extract a frame at 1 second
    const ffmpegProcess = spawn('ffmpeg', [
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
    
    // Use Electron's nativeImage to resize the image (this worked in old UI)
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

// IPC handlers
ipcMain.handle("select-files", handleFileImport)
ipcMain.handle("select-folder", handleFolderImport)

// App event handlers
app.whenReady().then(async () => {
  // Remove the killProcessesOnPorts call that might be causing issues
  // The npm script already handles port cleanup
  
  createWindow()
  
  // Add delay between starting services to prevent conflicts
  setTimeout(() => {
    startSearchServer()  // Start search server first
  }, 1000)
  
  setTimeout(() => {
    startPythonProcess() // Then start the unified service
  }, 3000)

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
