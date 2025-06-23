"use client"

import { useState, useEffect, useRef } from "react"
import { Upload, File, FolderOpen, AlertCircle, CheckCircle, Clock, X, StopCircle, FileText } from "lucide-react"
import { Button } from "@/components/ui/button"
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card"
import { Progress } from "@/components/ui/progress"
import { ScrollArea } from "@/components/ui/scroll-area"
import { Badge } from "@/components/ui/badge"
import { useToast } from "@/hooks/use-toast"

interface LogEntry {
  id: string
  timestamp: string
  type: 'info' | 'progress' | 'success' | 'error'
  message: string
  progress?: number
  stage?: string
}

// Uses the global electronAPI interface defined in dashboard-tab.tsx

export function UploadTab() {
  const [isProcessing, setIsProcessing] = useState(false)
  const [currentProgress, setCurrentProgress] = useState(0)
  const [logs, setLogs] = useState<LogEntry[]>([])
  const [processingStatus, setProcessingStatus] = useState<'idle' | 'processing' | 'success' | 'error' | 'stopped'>('idle')
  const logEndRef = useRef<HTMLDivElement>(null)
  const { toast } = useToast()

  // Counter to ensure unique log IDs
  const logCounterRef = useRef(0)

  // Load logs from localStorage on component mount
  useEffect(() => {
    if (typeof window !== 'undefined') {
      try {
        const savedLogs = localStorage.getItem('contentcache-processing-logs')
        const savedProgress = localStorage.getItem('contentcache-processing-progress')
        const savedStatus = localStorage.getItem('contentcache-processing-status')
        
        if (savedLogs) {
          const parsedLogs = JSON.parse(savedLogs)
          setLogs(parsedLogs)
          logCounterRef.current = parsedLogs.length
        }
        
        if (savedProgress) {
          setCurrentProgress(Number(savedProgress))
        }
        
        if (savedStatus && savedStatus !== 'idle') {
          setProcessingStatus(savedStatus as any)
          // If we're restoring a processing status, set isProcessing appropriately
          if (savedStatus === 'processing') {
            setIsProcessing(true)
          }
        }
      } catch (error) {
        console.error('Error loading saved logs:', error)
      }
    }
  }, [])

  // Save logs to localStorage whenever logs change
  useEffect(() => {
    if (typeof window !== 'undefined' && logs.length > 0) {
      try {
        localStorage.setItem('contentcache-processing-logs', JSON.stringify(logs))
      } catch (error) {
        console.error('Error saving logs:', error)
      }
    }
  }, [logs])

  // Save progress and status to localStorage
  useEffect(() => {
    if (typeof window !== 'undefined') {
      localStorage.setItem('contentcache-processing-progress', currentProgress.toString())
    }
  }, [currentProgress])

  useEffect(() => {
    if (typeof window !== 'undefined') {
      localStorage.setItem('contentcache-processing-status', processingStatus)
    }
  }, [processingStatus])

  // Auto-scroll to bottom when new logs are added
  useEffect(() => {
    logEndRef.current?.scrollIntoView({ behavior: "smooth" })
  }, [logs])

  const addLog = (type: LogEntry['type'], message: string, progress?: number, stage?: string) => {
    const newLog: LogEntry = {
      id: `${Date.now()}-${++logCounterRef.current}`,
      timestamp: new Date().toLocaleTimeString(),
      type,
      message,
      progress,
      stage
    }
    setLogs(prev => [...prev, newLog])
  }

  useEffect(() => {
    if (typeof window !== "undefined" && window.electronAPI) {
      // Listen for file selection events
      const cleanupFilesSelected = window.electronAPI.onFilesSelected((filePaths) => {
        console.log("DEBUG: Files selected event:", filePaths)
        handleFilesSelected(filePaths)
      })

      const cleanupFolderSelected = window.electronAPI.onFolderSelected((folderPath) => {
        console.log("DEBUG: Folder selected event:", folderPath)
        handleFolderSelected(folderPath)
      })

      const cleanupPythonOutput = window.electronAPI.onPythonOutput((data) => {
        try {
          const response = JSON.parse(data)

          switch (response.type) {
            case "progress":
              setCurrentProgress(response.progress)
              addLog('progress', `${response.stage}: ${response.message}`, response.progress, response.stage)
              break

            case "batch_progress":
              setCurrentProgress(response.progress)
              addLog('progress', `Processing ${response.completed}/${response.total} files (${response.progress.toFixed(1)}%)`, response.progress)
              break

            case "processing_complete":
              setIsProcessing(false)
              setProcessingStatus(response.successful > 0 ? 'success' : 'error')
              setCurrentProgress(100)
              addLog('success', `Processing complete! Successfully processed ${response.successful}/${response.total_processed} files`)
              
              // Clear localStorage processing state since processing is complete
              if (typeof window !== 'undefined') {
                localStorage.removeItem('contentcache-processing-progress')
                localStorage.removeItem('contentcache-processing-status')
              }
              
              toast({
                title: "Processing Complete!",
                description: `Successfully processed ${response.successful}/${response.total_processed} files`,
              })
              break

            case "system_status":
              addLog('info', `System status: ${response.search_server?.running ? 'Search server online' : 'Search server offline'}`)
              break

            default:
              if (response.message) {
                addLog('info', response.message)
              }
          }
        } catch (error) {
          // Handle non-JSON output (plain text messages)
          if (data.startsWith("OUTPUT:")) {
            const message = data.replace("OUTPUT:", "").trim()
            addLog('info', message)
          } else {
            addLog('info', data.trim())
          }
        }
      })

      const cleanupPythonError = window.electronAPI.onPythonError((data) => {
        addLog('error', `Error: ${data.trim()}`)
        setProcessingStatus('error')
      })

      const cleanupSearchServerOutput = window.electronAPI.onSearchServerOutput((data) => {
        // Filter out verbose logs, only show important ones
        if (data.includes('Ready to search') || data.includes('Loading') || data.includes('✅') || data.includes('🚀')) {
          addLog('info', `Search Server: ${data.trim()}`)
        }
      })

      const cleanupSearchServerError = window.electronAPI.onSearchServerError((data) => {
        if (!data.includes('INFO:') && !data.includes('WARNING:')) {
          addLog('error', `Search Server Error: ${data.trim()}`)
        }
      })

      return () => {
        cleanupFilesSelected()
        cleanupFolderSelected()
        cleanupPythonOutput()
        cleanupPythonError()
        cleanupSearchServerOutput()
        cleanupSearchServerError()
      }
    }
  }, [toast])

  const handleFilesSelected = async (filePaths: string[]) => {
    setIsProcessing(true)
    setProcessingStatus('processing')
    setCurrentProgress(0)
    addLog('info', `Starting to process ${filePaths.length} file(s)`)
    
    // Save processing state to localStorage
    if (typeof window !== 'undefined') {
      localStorage.setItem('contentcache-processing-status', 'processing')
      localStorage.setItem('contentcache-processing-progress', '0')
    }
    
    try {
      await window.electronAPI.processFiles(filePaths)
    } catch (error) {
      setProcessingStatus('error')
      addLog('error', `Failed to process files: ${error}`)
      toast({
        title: "Error",
        description: "Failed to process files",
        variant: "destructive",
      })
    }
  }

  const handleFolderSelected = async (folderPath: string) => {
    setIsProcessing(true)
    setProcessingStatus('processing')
    setCurrentProgress(0)
    addLog('info', `Starting to process folder: ${folderPath}`)

    // Save processing state to localStorage
    if (typeof window !== 'undefined') {
      localStorage.setItem('contentcache-processing-status', 'processing')
      localStorage.setItem('contentcache-processing-progress', '0')
    }

    try {
      await window.electronAPI.processFiles([folderPath])
    } catch (error) {
      setProcessingStatus('error')
      addLog('error', `Failed to process folder: ${error}`)
      toast({
        title: "Error",
        description: "Failed to process folder",
        variant: "destructive",
      })
    }
  }

  const handleSelectFiles = async () => {
    if (window.electronAPI) {
      await window.electronAPI.selectFiles()
    }
  }

  const handleSelectFolder = async () => {
    if (window.electronAPI) {
      await window.electronAPI.selectFolder()
    }
  }

  const clearLogs = () => {
    setLogs([])
    setProcessingStatus('idle')
    setCurrentProgress(0)
    logCounterRef.current = 0
    
    // Clear localStorage
    if (typeof window !== 'undefined') {
      localStorage.removeItem('contentcache-processing-logs')
      localStorage.removeItem('contentcache-processing-progress')
      localStorage.removeItem('contentcache-processing-status')
    }
  }

  const getStatusIcon = () => {
    switch (processingStatus) {
      case 'processing':
        return <Clock className="w-5 h-5 text-blue-500 animate-spin" />
      case 'success':
        return <CheckCircle className="w-5 h-5 text-green-500" />
      case 'error':
        return <AlertCircle className="w-5 h-5 text-red-500" />
      default:
        return <Upload className="w-5 h-5 text-gray-500" />
    }
  }

  const getLogIcon = (type: LogEntry['type']) => {
    switch (type) {
      case 'success':
        return <CheckCircle className="w-4 h-4 text-green-500" />
      case 'error':
        return <AlertCircle className="w-4 h-4 text-red-500" />
      case 'progress':
        return <Clock className="w-4 h-4 text-blue-500" />
      default:
        return <div className="w-4 h-4 rounded-full bg-gray-400" />
    }
  }

  const stopProcessing = async () => {
    if (window.electronAPI && window.electronAPI.stopProcessing) {
      try {
        await window.electronAPI.stopProcessing()
        setIsProcessing(false)
        setProcessingStatus('stopped')
        addLog('info', 'Processing stopped by user')
        toast({
          title: "Processing Stopped",
          description: "File processing has been stopped",
        })
      } catch (error) {
        addLog('error', `Failed to stop processing: ${error}`)
        toast({
          title: "Error",
          description: "Failed to stop processing",
          variant: "destructive",
        })
      }
    }
  }

  // Debug function to show metadata paths
  const showMetadataPaths = async () => {
    if (window.electronAPI) {
      try {
        // Request metadata paths from main process using generic invoke
        const paths = await (window.electronAPI as any).getMetadataPaths()
        addLog('info', `📁 Metadata Paths:`)
        addLog('info', `Video: ${paths.video}`)
        addLog('info', `Audio: ${paths.audio}`)
        addLog('info', `Text: ${paths.text}`)
        addLog('info', `Image: ${paths.image}`)
        toast({
          title: "Metadata Paths",
          description: "Check logs for full paths",
        })
      } catch (error) {
        addLog('error', `Failed to get metadata paths: ${error}`)
      }
    }
  }

  // Test API connectivity function
  const testApiConnectivity = async () => {
    if (window.electronAPI) {
      try {
        addLog('info', '🧪 Testing API connectivity...')
        // Use the generic process file function to run the API test
        const result = await (window.electronAPI as any).testApiConnectivity()
        if (result.status === 'success') {
          addLog('success', `✅ API Test Passed: ${result.message}`)
          addLog('info', `🌐 API URL: ${result.api_url}`)
          toast({
            title: "API Test Success",
            description: "Railway API server is accessible and working",
          })
        } else {
          addLog('error', `❌ API Test Failed: ${result.message}`)
          toast({
            title: "API Test Failed",
            description: "Check logs for details",
            variant: "destructive",
          })
        }
      } catch (error) {
        addLog('error', `Failed to test API: ${error}`)
        toast({
          title: "API Test Error",
          description: "Failed to run API test",
          variant: "destructive",
        })
      }
    }
  }

  return (
    <div className="flex-1 space-y-6 p-8">
      {/* Header */}
      <div>
        <h1 className="text-4xl font-bold text-gray-900 mb-2">Upload & Process Content</h1>
        <p className="text-lg text-gray-600">Add new files and folders to your content library with AI-powered analysis</p>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        {/* Upload Section */}
        <div className="space-y-6">
          {/* File Upload Card */}
          <Card className="border-2 border-dashed border-blue-200 bg-blue-50/30 hover:bg-blue-50/50 transition-colors">
            <CardContent className="p-8">
              <div className="text-center space-y-4">
                <div className="w-16 h-16 bg-blue-600 rounded-2xl flex items-center justify-center mx-auto">
                  <Upload className="w-8 h-8 text-white" />
                </div>
                <div>
                  <h3 className="text-xl font-semibold text-gray-900 mb-2">Upload Files</h3>
                  <p className="text-gray-600 mb-4">Select individual files or entire folders for processing</p>
                </div>
                <div className="flex flex-col sm:flex-row gap-4">
                  <Button
                    variant="outline"
                    size="lg"
                    className="px-8 py-3 text-base font-medium border-blue-200 hover:bg-blue-50"
                    onClick={handleSelectFiles}
                    disabled={isProcessing}
                  >
                    <FileText className="w-5 h-5 mr-2" />
                    Select Files
                  </Button>
                  <Button
                    variant="outline"
                    size="lg"
                    className="px-8 py-3 text-base font-medium border-blue-200 hover:bg-blue-50"
                    onClick={handleSelectFolder}
                    disabled={isProcessing}
                  >
                    <FolderOpen className="w-5 h-5 mr-2" />
                    Select Folder
                  </Button>
                </div>
                <div className="flex flex-wrap gap-2">
                  <Button
                    variant="outline"
                    size="sm"
                    className="px-3 py-2 text-xs border-gray-300 hover:bg-gray-50"
                    onClick={showMetadataPaths}
                    title="Debug: Show metadata file paths"
                  >
                    🐛 Paths
                  </Button>
                  <Button
                    variant="outline"
                    size="sm"
                    className="px-3 py-2 text-xs border-blue-300 hover:bg-blue-50"
                    onClick={testApiConnectivity}
                    title="Test Railway API connectivity"
                  >
                    🧪 API
                  </Button>
                </div>
                <p className="text-sm text-gray-500">Supports: Videos (MP4, MOV, AVI, MKV), Audio (MP3, WAV, AAC, FLAC), Images (JPG, PNG, HEIC), Text (TXT, PDF, MD)</p>
              </div>
            </CardContent>
          </Card>

          {/* Processing Status */}
          {(isProcessing || processingStatus !== 'idle') && (
            <Card>
              <CardHeader className="pb-3">
                <div className="flex items-center justify-between">
                  <div className="flex items-center gap-2">
                    {getStatusIcon()}
                    <CardTitle className="text-lg">Processing Status</CardTitle>
                  </div>
                  <Badge variant={processingStatus === 'success' ? 'default' : processingStatus === 'error' ? 'destructive' : 'secondary'}>
                    {processingStatus === 'processing' ? 'In Progress' : processingStatus === 'success' ? 'Complete' : processingStatus === 'error' ? 'Error' : 'Ready'}
                  </Badge>
                </div>
              </CardHeader>
              <CardContent className="space-y-4">
                {isProcessing && (
                  <>
                    <div className="space-y-2">
                      <div className="flex justify-between text-sm">
                        <span>Progress</span>
                        <span>{currentProgress.toFixed(1)}%</span>
                      </div>
                      <Progress value={currentProgress} className="w-full" />
                    </div>
                    <div className="flex justify-center">
                      <Button
                        variant="destructive"
                        size="sm"
                        onClick={stopProcessing}
                        className="flex items-center gap-2"
                      >
                        <StopCircle className="w-4 h-4" />
                        Stop Processing
                      </Button>
                    </div>
                  </>
                )}
              </CardContent>
            </Card>
          )}
        </div>

        {/* Logging Output */}
        <Card className="lg:row-span-2">
          <CardHeader className="pb-3">
            <div className="flex items-center justify-between">
              <div>
                <CardTitle className="text-lg">Processing Logs</CardTitle>
                <CardDescription>Real-time output from the processing pipeline</CardDescription>
              </div>
              <Button
                variant="outline"
                size="sm"
                onClick={clearLogs}
                disabled={isProcessing}
              >
                <X className="w-4 h-4 mr-1" />
                Clear
              </Button>
            </div>
          </CardHeader>
          <CardContent>
            <ScrollArea className="h-96 border rounded-md p-4 bg-gray-50">
              {logs.length === 0 ? (
                <div className="text-center text-gray-500 py-8">
                  <Upload className="w-8 h-8 mx-auto mb-2 opacity-50" />
                  <p>Processing logs will appear here</p>
                </div>
              ) : (
                <div className="space-y-2">
                  {logs.map((log) => (
                    <div key={log.id} className="flex items-start gap-2 text-sm">
                      <div className="mt-0.5">{getLogIcon(log.type)}</div>
                      <div className="flex-1">
                        <div className="flex items-center gap-2">
                          <span className="text-gray-500 text-xs">{log.timestamp}</span>
                          {log.stage && (
                            <Badge variant="outline" className="text-xs">
                              {log.stage}
                            </Badge>
                          )}
                          {log.progress !== undefined && (
                            <Badge variant="secondary" className="text-xs">
                              {log.progress.toFixed(1)}%
                            </Badge>
                          )}
                        </div>
                        <p className={`mt-1 ${log.type === 'error' ? 'text-red-700' : log.type === 'success' ? 'text-green-700' : 'text-gray-700'}`}>
                          {log.message}
                        </p>
                      </div>
                    </div>
                  ))}
                  <div ref={logEndRef} />
                </div>
              )}
            </ScrollArea>
          </CardContent>
        </Card>
      </div>
    </div>
  )
} 