#!/usr/bin/env node

const fs = require('fs')
const path = require('path')
const { spawn, exec } = require('child_process')
const https = require('https')
const http = require('http')

const PYTHON_VERSION = '3.11.9'
const PLATFORM = process.platform
const ARCH = process.arch

console.log('🐍 Bundling Python for distribution...')

// Create scripts directory if it doesn't exist
const scriptsDir = path.dirname(__filename)
if (!fs.existsSync(scriptsDir)) {
  fs.mkdirSync(scriptsDir, { recursive: true })
}

// Create python-dist directory
const pythonDistDir = path.join(__dirname, '..', 'python-dist')
if (!fs.existsSync(pythonDistDir)) {
  fs.mkdirSync(pythonDistDir, { recursive: true })
}

async function downloadFile(url, destination) {
  return new Promise((resolve, reject) => {
    console.log(`📥 Downloading ${url}`)
    const file = fs.createWriteStream(destination)
    
    const client = url.startsWith('https:') ? https : http
    
    client.get(url, (response) => {
      // Handle redirects
      if (response.statusCode >= 300 && response.statusCode < 400 && response.headers.location) {
        file.close()
        fs.unlinkSync(destination)
        downloadFile(response.headers.location, destination).then(resolve).catch(reject)
        return
      }
      
      if (response.statusCode !== 200) {
        file.close()
        fs.unlinkSync(destination)
        reject(new Error(`HTTP ${response.statusCode}: ${response.statusMessage}`))
        return
      }
      
      response.pipe(file)
      file.on('finish', () => {
        file.close()
        console.log(`✅ Downloaded to ${destination}`)
        resolve()
      })
    }).on('error', (err) => {
      file.close()
      fs.unlink(destination, () => {}) // Delete the file on error
      reject(err)
    })
  })
}

async function extractArchive(archivePath, extractDir) {
  return new Promise((resolve, reject) => {
    console.log(`📂 Extracting ${archivePath}`)
    
    let cmd
    if (archivePath.endsWith('.tar.gz') || archivePath.endsWith('.tgz')) {
      cmd = `tar -xzf "${archivePath}" -C "${extractDir}"`
    } else if (archivePath.endsWith('.zip')) {
      cmd = `unzip -q "${archivePath}" -d "${extractDir}"`
    } else {
      reject(new Error('Unsupported archive format'))
      return
    }
    
    exec(cmd, (error, stdout, stderr) => {
      if (error) {
        reject(error)
      } else {
        console.log(`✅ Extracted to ${extractDir}`)
        resolve()
      }
    })
  })
}

async function installPythonPackages(pythonExe, requirementsPath) {
  return new Promise((resolve, reject) => {
    console.log(`📦 Installing Python packages with --user flag for packaged app...`)
    
    // Use --user flag to install packages in user directory that packaged apps can access
    const installProcess = spawn(pythonExe, ['-m', 'pip', 'install', '--user', '-r', requirementsPath], {
      stdio: 'inherit'
    })
    
    installProcess.on('close', (code) => {
      if (code === 0) {
        console.log('✅ Python packages installed successfully in user directory')
        resolve()
      } else {
        reject(new Error(`pip install failed with code ${code}`))
      }
    })
    
    installProcess.on('error', (error) => {
      reject(error)
    })
  })
}

async function copyBackendModules() {
  console.log('📋 Copying backend processing modules...')
  
  const sourceBackendDir = path.join(__dirname, '..', 'backend')
  const destBackendDir = path.join(pythonDistDir, 'backend')
  
  if (!fs.existsSync(sourceBackendDir)) {
    console.log('⚠️ Backend source directory not found, skipping backend module copy')
    return
  }
  
  // Copy the entire backend directory
  await copyDirectory(sourceBackendDir, destBackendDir)
  console.log('✅ Backend modules copied successfully')
}

async function copyDirectory(source, destination) {
  if (!fs.existsSync(destination)) {
    fs.mkdirSync(destination, { recursive: true })
  }
  
  const items = fs.readdirSync(source)
  
  for (const item of items) {
    const sourcePath = path.join(source, item)
    const destPath = path.join(destination, item)
    
    const stat = fs.statSync(sourcePath)
    
    if (stat.isDirectory()) {
      await copyDirectory(sourcePath, destPath)
    } else {
      fs.copyFileSync(sourcePath, destPath)
    }
  }
}

async function downloadFFmpeg() {
  console.log('📥 Downloading ffmpeg binary...')
  
  const ffmpegDir = path.join(__dirname, '..', 'binaries')
  const ffmpegPath = path.join(ffmpegDir, 'ffmpeg')
  const ffprobePath = path.join(ffmpegDir, 'ffprobe')
  
  // Create binaries directory
  if (!fs.existsSync(ffmpegDir)) {
    fs.mkdirSync(ffmpegDir, { recursive: true })
  }
  
  // Check if already downloaded
  if (fs.existsSync(ffmpegPath) && fs.existsSync(ffprobePath)) {
    console.log('✅ FFmpeg binaries already exist')
    return
  }
  
  try {
    if (PLATFORM === 'darwin') {
      // Download ffmpeg static build for macOS
      const ffmpegZipPath = path.join(ffmpegDir, 'ffmpeg.zip')
      const ffprobeZipPath = path.join(ffmpegDir, 'ffprobe.zip')
      
      console.log('🔄 Downloading ffmpeg...')
      await downloadFile('https://evermeet.cx/ffmpeg/ffmpeg-6.1.zip', ffmpegZipPath)
      
      console.log('🔄 Downloading ffprobe...')
      await downloadFile('https://evermeet.cx/ffmpeg/ffprobe-6.1.zip', ffprobeZipPath)
      
      // Extract using system unzip command
      console.log('📂 Extracting ffmpeg...')
      await new Promise((resolve, reject) => {
        exec(`unzip -o "${ffmpegZipPath}" -d "${ffmpegDir}"`, (error, stdout, stderr) => {
          if (error) {
            reject(error)
          } else {
            resolve()
          }
        })
      })
      
      console.log('📂 Extracting ffprobe...')
      await new Promise((resolve, reject) => {
        exec(`unzip -o "${ffprobeZipPath}" -d "${ffmpegDir}"`, (error, stdout, stderr) => {
          if (error) {
            reject(error)
          } else {
            resolve()
          }
        })
      })
      
      // Clean up zip files
      fs.unlinkSync(ffmpegZipPath)
      fs.unlinkSync(ffprobeZipPath)
      
      // Make executable
      fs.chmodSync(ffmpegPath, 0o755)
      fs.chmodSync(ffprobePath, 0o755)
      
      console.log('✅ FFmpeg binaries downloaded and extracted')
      
    } else if (PLATFORM === 'win32') {
      // For Windows, download from a different source
      const ffmpegZipPath = path.join(ffmpegDir, 'ffmpeg-win.zip')
      
      console.log('🔄 Downloading ffmpeg for Windows...')
      await downloadFile('https://www.gyan.dev/ffmpeg/builds/release/ffmpeg-release-essentials.zip', ffmpegZipPath)
      
      // Extract using system command
      console.log('📂 Extracting ffmpeg...')
      await new Promise((resolve, reject) => {
        exec(`powershell Expand-Archive -Path "${ffmpegZipPath}" -DestinationPath "${ffmpegDir}" -Force`, (error) => {
          if (error) {
            reject(error)
          } else {
            resolve()
          }
        })
      })
      
      // Clean up
      fs.unlinkSync(ffmpegZipPath)
      
      console.log('✅ FFmpeg binaries downloaded and extracted for Windows')
      
    } else {
      // For Linux, download static builds
      const ffmpegTarPath = path.join(ffmpegDir, 'ffmpeg-linux.tar.xz')
      
      console.log('🔄 Downloading ffmpeg for Linux...')
      await downloadFile('https://johnvansickle.com/ffmpeg/releases/ffmpeg-release-amd64-static.tar.xz', ffmpegTarPath)
      
      // Extract using system command
      console.log('📂 Extracting ffmpeg...')
      await new Promise((resolve, reject) => {
        exec(`tar -xf "${ffmpegTarPath}" -C "${ffmpegDir}" --strip-components=1`, (error) => {
          if (error) {
            reject(error)
          } else {
            resolve()
          }
        })
      })
      
      // Clean up
      fs.unlinkSync(ffmpegTarPath)
      
      console.log('✅ FFmpeg binaries downloaded and extracted for Linux')
    }
    
  } catch (error) {
    console.log('⚠️ Warning: Could not download ffmpeg binaries:', error.message)
    console.log('   Video frame extraction may not work without system ffmpeg')
  }
}

async function bundlePython() {
  try {
    let pythonUrl, pythonExe, pythonDir
    
    if (PLATFORM === 'darwin') {
      // Use Python.org official distribution for macOS
      const macArch = ARCH === 'arm64' ? 'arm64' : 'intel'
      pythonUrl = `https://www.python.org/ftp/python/${PYTHON_VERSION}/python-${PYTHON_VERSION}-macos11.pkg`
      
      // For macOS, we'll use the system Python and just bundle packages
      console.log('📋 Using system Python for macOS (recommended approach)')
      pythonExe = 'python3'
      
    } else if (PLATFORM === 'win32') {
      // Use Python embedded distribution for Windows
      pythonUrl = `https://www.python.org/ftp/python/${PYTHON_VERSION}/python-${PYTHON_VERSION}-embed-amd64.zip`
      const archivePath = path.join(pythonDistDir, 'python-windows.zip')
      
      await downloadFile(pythonUrl, archivePath)
      await extractArchive(archivePath, pythonDistDir)
      
      pythonExe = path.join(pythonDistDir, 'python.exe')
      
    } else {
      // Linux - use portable Python
      pythonUrl = `https://github.com/indygreg/python-build-standalone/releases/download/20231002/cpython-${PYTHON_VERSION}+20231002-x86_64-unknown-linux-gnu-install_only.tar.gz`
      const archivePath = path.join(pythonDistDir, 'python-linux.tar.gz')
      
      await downloadFile(pythonUrl, archivePath)
      await extractArchive(archivePath, pythonDistDir)
      
      pythonExe = path.join(pythonDistDir, 'python', 'bin', 'python3')
    }
    
    // Copy backend processing modules
    await copyBackendModules()
    
    // Download ffmpeg binaries
    await downloadFFmpeg()
    
    // Skip Python package installation during build - the app will handle it on first run
    console.log('📦 Skipping Python package installation during build time')
    console.log('   Packages will be installed automatically when user first runs the app')
    
    console.log('🎉 Python bundling complete!')
    
  } catch (error) {
    console.error('❌ Python bundling failed:', error.message)
    process.exit(1)
  }
}

// Run the bundling process
bundlePython() 