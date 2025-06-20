const { app, BrowserWindow, ipcMain, dialog } = require('electron');
const path = require('path');
const { spawn } = require('child_process');

let mainWindow;
let searchServerProcess;

function createWindow() {
  mainWindow = new BrowserWindow({
    width: 1400,
    height: 900,
    webPreferences: {
      nodeIntegration: true,
      contextIsolation: false
    }
  });

  mainWindow.loadFile('index.html');
  
  // Open DevTools in development
  if (process.argv.includes('--dev')) {
    mainWindow.webContents.openDevTools();
  }
  
  // Start search server
  startSearchServer();
}

function startSearchServer() {
  console.log('🚀 Starting search server...');
  
  searchServerProcess = spawn('python', ['search_server.py'], {
    cwd: __dirname,
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
  
  // Wait a moment for server to start
  setTimeout(() => {
    console.log('✅ Search server should be running on http://localhost:5001');
  }, 3000);
}

app.whenReady().then(createWindow);

app.on('window-all-closed', () => {
  // Kill search server when app closes
  if (searchServerProcess) {
    searchServerProcess.kill();
  }
  
  if (process.platform !== 'darwin') {
    app.quit();
  }
});

app.on('activate', () => {
  if (BrowserWindow.getAllWindows().length === 0) {
    createWindow();
  }
});

// IPC handlers for video processing
ipcMain.handle('select-video-file', async () => {
  const result = await dialog.showOpenDialog(mainWindow, {
    properties: ['openFile'],
    filters: [
      { name: 'Video Files', extensions: ['mp4', 'mov', 'avi', 'mkv'] }
    ]
  });
  
  return result.filePaths[0];
});

ipcMain.handle('process-video', async (event, videoPath, options = {}) => {
  return new Promise((resolve, reject) => {
    const args = ['-c', `python videotagger.py "${videoPath}"`];
    
    if (options.concurrent) {
      args[1] += ' --concurrent';
    }
    if (options.api) {
      args[1] += ' --api';
    }
    
    const pythonProcess = spawn('/bin/zsh', args, {
      cwd: __dirname
    });
    
    let output = '';
    let error = '';
    
    pythonProcess.stdout.on('data', (data) => {
      output += data.toString();
      // Send progress updates to renderer
      mainWindow.webContents.send('processing-progress', data.toString());
    });
    
    pythonProcess.stderr.on('data', (data) => {
      error += data.toString();
      mainWindow.webContents.send('processing-error', data.toString());
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