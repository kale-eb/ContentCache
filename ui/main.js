const { app, BrowserWindow, ipcMain, dialog, nativeImage } = require('electron');
const path = require('path');
const { spawn } = require('child_process');
const fs = require('fs');
const os = require('os');

// Keep a global reference of the window object
let mainWindow;
let searchServerProcess = null;

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
      preload: path.join(__dirname, 'preload.js')
    },
    titleBarStyle: 'hiddenInset',
    show: false,
    backgroundColor: '#1a1a1a'
  });

  // Load the app
  mainWindow.loadFile('index.html');

  // Show window when ready to prevent visual flash
  mainWindow.once('ready-to-show', () => {
    mainWindow.show();
  });

  // Emitted when the window is closed
  mainWindow.on('closed', () => {
    mainWindow = null;
  });

  // Open DevTools in development
  if (process.env.NODE_ENV === 'development') {
    mainWindow.webContents.openDevTools();
  }
}

// Start the search server
function startSearchServer() {
  const searchServerPath = path.join(__dirname, '..', 'backend', 'search', 'search_server.py');
  
  if (fs.existsSync(searchServerPath)) {
    console.log('Starting search server...');
    searchServerProcess = spawn('python', [searchServerPath, '--port', '5001'], {
      cwd: path.join(__dirname, '..', 'backend', 'search'),
      stdio: ['pipe', 'pipe', 'pipe']
    });

    searchServerProcess.stdout.on('data', (data) => {
      console.log(`Search Server: ${data}`);
    });

    searchServerProcess.stderr.on('data', (data) => {
      console.error(`Search Server Error: ${data}`);
    });

    searchServerProcess.on('close', (code) => {
      console.log(`Search server exited with code ${code}`);
    });

    searchServerProcess.on('error', (error) => {
      console.error(`Failed to start search server: ${error}`);
    });
  } else {
    console.error(`Search server script not found at: ${searchServerPath}`);
  }
}

// Stop the search server
function stopSearchServer() {
  if (searchServerProcess) {
    searchServerProcess.kill();
    searchServerProcess = null;
  }
}

// App event handlers
app.whenReady().then(() => {
  createWindow();
  startSearchServer();
  
  // Give the search server time to start up before the UI tries to connect
  setTimeout(() => {
    if (mainWindow) {
      mainWindow.webContents.send('search-server-starting');
    }
  }, 3000);

  app.on('activate', () => {
    if (BrowserWindow.getAllWindows().length === 0) {
      createWindow();
    }
  });
});

app.on('window-all-closed', () => {
  stopSearchServer();
  if (process.platform !== 'darwin') {
    app.quit();
  }
});

app.on('before-quit', () => {
  stopSearchServer();
});

// IPC handlers for unified service communication
ipcMain.handle('select-directory', async () => {
  const result = await dialog.showOpenDialog(mainWindow, {
    properties: ['openDirectory'],
    title: 'Select Directory to Process'
  });
  
  if (!result.canceled && result.filePaths.length > 0) {
    return result.filePaths[0];
  }
  return null;
});

ipcMain.handle('select-file', async () => {
  const result = await dialog.showOpenDialog(mainWindow, {
    properties: ['openFile'],
    title: 'Select File to Process',
    filters: [
      { name: 'All Supported', extensions: ['mp4', 'mov', 'avi', 'mkv', 'jpg', 'jpeg', 'png', 'pdf', 'txt', 'md', 'mp3', 'wav'] },
      { name: 'Videos', extensions: ['mp4', 'mov', 'avi', 'mkv', 'wmv', 'flv', 'webm'] },
      { name: 'Images', extensions: ['jpg', 'jpeg', 'png', 'bmp', 'tiff', 'webp', 'heic'] },
      { name: 'Documents', extensions: ['pdf', 'txt', 'md', 'docx', 'rtf'] },
      { name: 'Audio', extensions: ['mp3', 'wav', 'm4a', 'flac', 'aac', 'ogg'] }
    ]
  });
  
  if (!result.canceled && result.filePaths.length > 0) {
    return result.filePaths[0];
  }
  return null;
});

ipcMain.handle('process-file', async (event, filePath) => {
  return new Promise((resolve, reject) => {
    const unifiedServicePath = path.join(__dirname, '..', 'backend', 'processing', 'unified_service.py');
    const pythonProcess = spawn('python', [unifiedServicePath, filePath], {
      cwd: path.join(__dirname, '..', 'backend', 'processing')
    });

    let output = '';
    let error = '';

    pythonProcess.stdout.on('data', (data) => {
      const chunk = data.toString();
      output += chunk;
      // Send progress updates to renderer
      event.sender.send('processing-progress', chunk);
    });

    pythonProcess.stderr.on('data', (data) => {
      error += data.toString();
    });

    pythonProcess.on('close', (code) => {
      if (code === 0) {
        resolve({ success: true, output });
      } else {
        reject({ success: false, error, code });
      }
    });
  });
});

ipcMain.handle('process-directory', async (event, directoryPath) => {
  return new Promise((resolve, reject) => {
    const unifiedServicePath = path.join(__dirname, '..', 'backend', 'processing', 'unified_service.py');
    const pythonProcess = spawn('python', [unifiedServicePath, directoryPath], {
      cwd: path.join(__dirname, '..', 'backend', 'processing')
    });

    let output = '';
    let error = '';

    pythonProcess.stdout.on('data', (data) => {
      const chunk = data.toString();
      output += chunk;
      // Send progress updates to renderer
      event.sender.send('processing-progress', chunk);
    });

    pythonProcess.stderr.on('data', (data) => {
      error += data.toString();
    });

    pythonProcess.on('close', (code) => {
      if (code === 0) {
        resolve({ success: true, output });
      } else {
        reject({ success: false, error, code });
      }
    });
  });
});

ipcMain.handle('search-content', async (event, query, contentTypes, topK) => {
  // Make HTTP request to search server
  const fetch = (await import('node-fetch')).default;
  
  try {
    const response = await fetch('http://127.0.0.1:5001/search', {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({
        query,
        type: contentTypes?.length === 1 ? contentTypes[0] : 'all',
        top_k: topK || 20
      })
    });

    if (response.ok) {
      return await response.json();
    } else {
      throw new Error(`Search server returned ${response.status}`);
    }
  } catch (error) {
    return {
      error: `Search failed: ${error.message}`,
      results: []
    };
  }
});

ipcMain.handle('get-search-status', async () => {
  const fetch = (await import('node-fetch')).default;
  
  try {
    const response = await fetch('http://127.0.0.1:5001/status', {
      method: 'GET'
    });

    if (response.ok) {
      const status = await response.json();
      return { running: true, ...status };
    } else {
      return { running: false, error: `Server returned ${response.status}` };
    }
  } catch (error) {
    return { running: false, error: error.message };
  }
});

ipcMain.handle('open-file', async (event, filePath) => {
  const { shell } = require('electron');
  
  try {
    // Check if file exists first
    if (!fs.existsSync(filePath)) {
      throw new Error(`File not found: ${filePath}`);
    }
    
    const result = await shell.openPath(filePath);
    
    // shell.openPath returns an error string if it fails, empty string if success
    if (result && result.length > 0) {
      throw new Error(`Failed to open file: ${result}`);
    }
    
    console.log(`Successfully opened: ${filePath}`);
    return { success: true };
  } catch (error) {
    console.error('Failed to open file:', error);
    return { success: false, error: error.message };
  }
});

ipcMain.handle('reveal-file', async (event, filePath) => {
  const { shell } = require('electron');
  
  try {
    // Check if file exists first
    if (!fs.existsSync(filePath)) {
      throw new Error(`File not found: ${filePath}`);
    }
    
    shell.showItemInFolder(filePath);
    console.log(`Successfully revealed: ${filePath}`);
    return { success: true };
  } catch (error) {
    console.error('Failed to reveal file:', error);
    return { success: false, error: error.message };
  }
});

// Thumbnail generation for video files
ipcMain.handle('generate-thumbnail', async (event, filePath, contentType) => {
  try {
    // Check if file exists
    if (!fs.existsSync(filePath)) {
      throw new Error(`File not found: ${filePath}`);
    }

    const ext = path.extname(filePath).toLowerCase();
    
    if (contentType === 'video' || ['.mp4', '.mov', '.avi', '.mkv', '.wmv', '.flv', '.webm'].includes(ext)) {
      return await generateVideoThumbnail(filePath);
    } else if (contentType === 'image' || ['.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.webp', '.heic'].includes(ext)) {
      return await generateImageThumbnail(filePath);
    } else {
      // Return a default icon for other file types
      return await generateDefaultThumbnail(contentType);
    }
  } catch (error) {
    console.error('Thumbnail generation failed:', error);
    return null;
  }
});

async function generateVideoThumbnail(videoPath) {
  return new Promise((resolve, reject) => {
    const tempDir = os.tmpdir();
    const thumbnailPath = path.join(tempDir, `thumb_${Date.now()}.jpg`);
    
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
    });

    ffmpegProcess.on('close', (code) => {
      if (code === 0 && fs.existsSync(thumbnailPath)) {
        try {
          // Read the thumbnail and convert to base64
          const thumbnailBuffer = fs.readFileSync(thumbnailPath);
          const base64Thumbnail = thumbnailBuffer.toString('base64');
          
          // Clean up temp file
          fs.unlinkSync(thumbnailPath);
          
          resolve(`data:image/jpeg;base64,${base64Thumbnail}`);
        } catch (error) {
          reject(error);
        }
      } else {
        // Fallback: try to use system thumbnail for video
        try {
          const systemThumbnail = app.getFileIcon(videoPath, { size: 'large' });
          systemThumbnail.then(icon => {
            resolve(icon.toDataURL());
          }).catch(() => resolve(null));
        } catch (error) {
          resolve(null);
        }
      }
    });

    ffmpegProcess.on('error', (error) => {
      // Fallback to system icon
      try {
        app.getFileIcon(videoPath, { size: 'large' }).then(icon => {
          resolve(icon.toDataURL());
        }).catch(() => resolve(null));
      } catch (e) {
        resolve(null);
      }
    });
  });
}

async function generateImageThumbnail(imagePath) {
  try {
    // Use Electron's nativeImage to resize the image
    const image = nativeImage.createFromPath(imagePath);
    if (image.isEmpty()) {
      return null;
    }
    
    // Resize to thumbnail width while maintaining aspect ratio
    const resized = image.resize({ width: 320 });
    
    return resized.toDataURL();
  } catch (error) {
    console.error('Image thumbnail generation failed:', error);
    
    // Fallback to system icon
    try {
      const icon = await app.getFileIcon(imagePath, { size: 'large' });
      return icon.toDataURL();
    } catch (e) {
      return null;
    }
  }
}

async function generateDefaultThumbnail(contentType) {
  // Generate a simple colored rectangle based on content type
  const colors = {
    text: '#3b82f6',    // Blue
    audio: '#f59e0b',   // Orange
    video: '#ef4444',   // Red
    image: '#10b981'    // Green
  };
  
  const color = colors[contentType] || '#6b7280';
  
  // Create a simple colored square as SVG and convert to data URL
  const svg = `
    <svg width="320" height="240" xmlns="http://www.w3.org/2000/svg">
      <rect width="320" height="240" fill="${color}" opacity="0.2"/>
      <text x="160" y="130" text-anchor="middle" fill="${color}" font-family="Arial" font-size="24" font-weight="bold">
        ${contentType.toUpperCase()}
      </text>
    </svg>
  `;
  
  return `data:image/svg+xml;base64,${Buffer.from(svg).toString('base64')}`;
}