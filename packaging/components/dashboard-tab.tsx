"use client"

import { useState, useEffect } from "react"
import { Search, Video, FileText, Upload, Filter, Grid, List, FolderOpen, File, ExternalLink, ChevronDown, ChevronUp, Calendar, MapPin } from "lucide-react"
import { useToast } from "@/hooks/use-toast"
import { Button } from "@/components/ui/button"
import { Input } from "@/components/ui/input"
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card"
import { Badge } from "@/components/ui/badge"
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select"

// Simple responsive card component
function SimpleCard({ 
  result, 
  isSelected, 
  onSelect, 
  onOpen, 
  onReveal 
}: { 
  result: any
  isSelected: boolean
  onSelect: (event: React.MouseEvent) => void
  onOpen: () => void
  onReveal: () => void
}) {
  const [thumbnailSrc, setThumbnailSrc] = useState<string>("/placeholder.svg")
  const [loading, setLoading] = useState(true)

  useEffect(() => {
    if (window.electronAPI && result.file_path) {
      window.electronAPI.generateThumbnail(result.file_path, result.content_type)
        .then((thumbnail: string | null) => {
          setThumbnailSrc(thumbnail || "/placeholder.svg")
          setLoading(false)
        })
        .catch(() => {
          setThumbnailSrc("/placeholder.svg")
          setLoading(false)
        })
    }
  }, [result.file_path, result.content_type])

  return (
    <div
      className={`group relative bg-white rounded-lg shadow-sm hover:shadow-md transition-shadow cursor-pointer border ${
        isSelected ? 'border-blue-500 ring-2 ring-blue-200' : 'border-gray-200 hover:border-gray-300'
      }`}
      onClick={onSelect}
    >
      {/* Thumbnail */}
      <div className="aspect-video bg-gray-100 rounded-t-lg overflow-hidden relative">
        {loading ? (
          <div className="w-full h-full flex items-center justify-center">
            <div className="w-5 h-5 border-2 border-gray-400 border-t-transparent rounded-full animate-spin" />
          </div>
        ) : (
          <img
            src={thumbnailSrc}
            alt={result.filename || 'Unknown file'}
            className="w-full h-full object-cover"
            onError={() => setThumbnailSrc("/placeholder.svg")}
          />
        )}

        {/* Selection indicator */}
        {isSelected && (
          <div className="absolute top-2 left-2 w-5 h-5 bg-blue-500 rounded-full flex items-center justify-center">
            <svg className="w-3 h-3 text-white" fill="currentColor" viewBox="0 0 20 20">
              <path fillRule="evenodd" d="M16.707 5.293a1 1 0 010 1.414l-8 8a1 1 0 01-1.414 0l-4-4a1 1 0 011.414-1.414L8 12.586l7.293-7.293a1 1 0 011.414 0z" clipRule="evenodd" />
            </svg>
          </div>
        )}

        {/* Action buttons - visible on hover */}
        <div className="absolute top-2 right-2 flex gap-1 opacity-0 group-hover:opacity-100 transition-opacity">
          <button
            onClick={(e) => { e.stopPropagation(); onOpen(); }}
            className="w-6 h-6 bg-black bg-opacity-70 hover:bg-opacity-90 text-white rounded-full flex items-center justify-center"
            title="Open file"
          >
            <ExternalLink className="w-3 h-3" />
          </button>
          <button
            onClick={(e) => { e.stopPropagation(); onReveal(); }}
            className="w-6 h-6 bg-black bg-opacity-70 hover:bg-opacity-90 text-white rounded-full flex items-center justify-center"
            title="Show in folder"
          >
            <FolderOpen className="w-3 h-3" />
          </button>
        </div>

        {/* Video duration */}
        {result.duration && (
          <div className="absolute bottom-2 right-2 bg-black bg-opacity-75 text-white text-xs px-1.5 py-0.5 rounded">
            {result.duration}
          </div>
        )}
      </div>

      {/* Content info */}
      <div className="p-3">
        <h3 className="font-medium text-sm text-gray-900 truncate mb-1">
          {result.filename || result.title || 'Unknown file'}
        </h3>
        <div className="flex items-center justify-between text-xs text-gray-500">
          <span className="bg-gray-100 px-2 py-1 rounded text-xs">
            {result.content_type}
          </span>
          <span>{(result.similarity_score * 100).toFixed(0)}%</span>
        </div>
      </div>
    </div>
  )
}



export function DashboardTab() {
  const [searchQuery, setSearchQuery] = useState("")
  const [selectedContentType, setSelectedContentType] = useState("all")
  const [viewMode, setViewMode] = useState<"grid" | "list">("grid")
  const [isProcessing, setIsProcessing] = useState(false)
  const [searchResults, setSearchResults] = useState<any[]>([])
  const [isSearching, setIsSearching] = useState(false)
  const [hasSearched, setHasSearched] = useState(false)
  const [searchTimeout, setSearchTimeout] = useState<NodeJS.Timeout | null>(null)
  const [isSearchInProgress, setIsSearchInProgress] = useState(false)
  
  // New filter states
  const [dateFilter, setDateFilter] = useState("")
  const [locationFilter, setLocationFilter] = useState("")
  const [showAdvancedFilters, setShowAdvancedFilters] = useState(false)
  
  // Bucket expansion states
  const [expandedBuckets, setExpandedBuckets] = useState<Set<string>>(new Set())
  
  const [stats, setStats] = useState({
    totalVideos: 0,
    totalImages: 0,
    totalText: 0,
    totalAudio: 0,
    storageUsed: "0 GB",
    searchServerStatus: false
  })
  const { toast } = useToast()

  // Remove hardcoded results - using real search results now

  // Fetch system status and stats (with rate limiting)
  const [lastFetchTime, setLastFetchTime] = useState(0)
  const fetchStats = async () => {
    const now = Date.now()
    if (now - lastFetchTime < 5000) { // Rate limit to once every 5 seconds
      return
    }
    
    if (window.electronAPI) {
      try {
        setLastFetchTime(now)
        await window.electronAPI.getSystemStatus()
      } catch (error) {
        console.error("Failed to fetch stats:", error)
      }
    }
  }

  useEffect(() => {
    if (typeof window !== "undefined" && window.electronAPI) {
      // Listen for file selection events
      const cleanupFilesSelected = window.electronAPI.onFilesSelected((filePaths: string[]) => {
        handleFilesSelected(filePaths)
      })

      const cleanupFolderSelected = window.electronAPI.onFolderSelected((folderPath: string) => {
        handleFolderSelected(folderPath)
      })

      const cleanupPythonOutput = window.electronAPI.onPythonOutput((data: string) => {
        // Safety check for undefined data
        if (data === undefined || data === null) {
          return
        }
        
        try {
          const response = JSON.parse(data)

          switch (response.type) {
            case "progress":
              toast({
                title: `${response.stage.toUpperCase()}`,
                description: `${response.progress.toFixed(1)}% - ${response.message}`,
              })
              break

            case "batch_progress":
              toast({
                title: "Processing Files",
                description: `${response.completed}/${response.total} files completed (${response.progress.toFixed(1)}%)`,
              })
              break

            case "processing_complete":
              setIsProcessing(false)
              toast({
                title: "Processing Complete!",
                description: `Successfully processed ${response.successful}/${response.total_processed} files`,
              })
              break

            case "search_results":
              if (searchTimeout) {
                clearTimeout(searchTimeout)
                setSearchTimeout(null)
              }
              setIsSearching(false)
              setHasSearched(true)
              setIsSearchInProgress(false)
              
              // Handle both bucketed and flat results from server
              if (response.has_buckets && response.buckets) {
                // Server returned bucketed results - preserve buckets but flatten for state
                const flatResults: any[] = []
                Object.entries(response.buckets).forEach(([bucketName, bucketResults]: [string, any]) => {
                  if (Array.isArray(bucketResults)) {
                    bucketResults.forEach((result: any) => {
                      flatResults.push({
                        ...result,
                        server_bucket: bucketName  // Track which server bucket this came from
                      })
                    })
                  }
                })
                setSearchResults(flatResults)
                
                toast({
                  title: "Search Complete",
                  description: `Found ${response.total_found || flatResults.length} results in ${Object.keys(response.buckets).length} categories for "${response.query}"`,
                })
              } else {
                // Server returned flat results
                setSearchResults(response.results || [])
              toast({
                title: "Search Complete",
                  description: `Found ${response.total_found || response.results?.length || 0} results for "${response.query}"`,
              })
              }
              break

            case "search_error":
              if (searchTimeout) {
                clearTimeout(searchTimeout)
                setSearchTimeout(null)
              }
              setIsSearching(false)
              setHasSearched(true)
              setIsSearchInProgress(false)
              setSearchResults([])
              toast({
                title: "Search Error",
                description: response.error,
                variant: "destructive",
              })
              break

            case "system_status":
              // Update stats from search server status
              if (response.search_server && response.search_server.stats) {
                setStats(prevStats => ({
                  ...prevStats,
                  totalVideos: response.search_server.stats.video || 0,
                  totalImages: response.search_server.stats.image || 0,
                  totalText: response.search_server.stats.text || 0,
                  totalAudio: response.search_server.stats.audio || 0,
                  searchServerStatus: response.search_server.running || false
                }))
              }
              
              if (!response.search_server?.running) {
                toast({
                  title: "Search Server Offline",
                  description: "Start the search server to enable content search",
                  variant: "destructive",
                })
              } else {
                const totalItems = (response.search_server.stats?.video || 0) + 
                                 (response.search_server.stats?.image || 0) + 
                                 (response.search_server.stats?.text || 0) + 
                                 (response.search_server.stats?.audio || 0)
                toast({
                  title: "Search Server Online",
                  description: `Ready to search ${totalItems} items`,
                })
              }
              break

            default:
              console.log("Python output:", response)
          }
        } catch (error) {
          // Handle non-JSON output (plain text messages)
          if (typeof data === 'string' && data.startsWith("OUTPUT:")) {
            const message = data.replace("OUTPUT:", "").trim()
            toast({
              title: "Processing Update",
              description: message,
            })
          }
        }
      })

      // Initial stats fetch
      fetchStats()

      return () => {
        cleanupFilesSelected()
        cleanupFolderSelected()
        cleanupPythonOutput()
      }
    }
  }, [toast])

  const handleFilesSelected = async (filePaths: string[]) => {
    setIsProcessing(true)
    toast({
      title: "Files Selected",
      description: `Processing ${filePaths.length} file(s)...`,
    })

    try {
      await window.electronAPI.processFiles(filePaths)
    } catch (error) {
      toast({
        title: "Error",
        description: "Failed to process files",
        variant: "destructive",
      })
    } finally {
      setIsProcessing(false)
    }
  }

  const handleFolderSelected = async (folderPath: string) => {
    setIsProcessing(true)
    toast({
      title: "Folder Selected",
      description: `Processing folder: ${folderPath}`,
    })

    try {
      await window.electronAPI.processFiles([folderPath])
    } catch (error) {
      toast({
        title: "Error",
        description: "Failed to process folder",
        variant: "destructive",
      })
    } finally {
      setIsProcessing(false)
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

  const [selectedItems, setSelectedItems] = useState<Set<string>>(new Set())
  const [draggedItem, setDraggedItem] = useState<any>(null)

  // Helper functions for date buckets
  const organizeBuckets = (results: any[]) => {
    // Check if results came from server-side buckets
    const hasServerBuckets = results.some(result => result.server_bucket)
    
    if (hasServerBuckets) {
      // Use server-side buckets - group by server_bucket field
      const buckets: { [key: string]: any[] } = {}
      results.forEach(result => {
        const bucketName = result.server_bucket || "Other Results"
        if (!buckets[bucketName]) {
          buckets[bucketName] = []
        }
        buckets[bucketName].push(result)
      })
      return buckets
    }
    
    // Fallback to client-side bucketing (for backward compatibility)
    const hasDateFilter = dateFilter.trim() !== ""
    const hasLocationFilter = locationFilter.trim() !== ""
    
    if (!hasDateFilter && !hasLocationFilter) {
      // No filters - return results as single flat list
      return { "All Results": results }
    }
    
    const buckets: { [key: string]: any[] } = {}
    
    // Helper function to check if a result matches date criteria
    const matchesDate = (result: any, targetDate: string) => {
      if (!targetDate) return false
      
      const extractDate = (result: any) => {
        if (result.date_recorded) {
          return new Date(result.date_recorded).toISOString().split('T')[0]
        } else if (result.processed_at) {
          return new Date(result.processed_at).toISOString().split('T')[0]
        } else if (result.file_path) {
          const dateMatch = result.file_path.match(/(\d{4})[_-]?(\d{2})[_-]?(\d{2})/)
          if (dateMatch) {
            const [, year, month, day] = dateMatch
            return `${year}-${month.padStart(2, '0')}-${day.padStart(2, '0')}`
          }
        }
        return null
      }
      
      const resultDate = extractDate(result)
      return resultDate && resultDate.includes(targetDate)
    }
    
    // Helper function to check if a result matches location criteria  
    const matchesLocation = (result: any, targetLocation: string) => {
      if (!targetLocation) return false
      
      const location = result.location || result.gps_location || ""
      return location.toLowerCase().includes(targetLocation.toLowerCase())
    }
    
    if (hasDateFilter && hasLocationFilter) {
      // Both date and location filters - create 4 buckets with ALL results
      const bucketKeys = ["📅📍 Date & Location Match", "📅 Date Match Only", "📍 Location Match Only", "📄 Other Results"]
      bucketKeys.forEach(key => { buckets[key] = [] })
      
      results.forEach(result => {
        const dateMatch = matchesDate(result, dateFilter)
        const locationMatch = matchesLocation(result, locationFilter)
        
        if (dateMatch && locationMatch) {
          buckets["📅📍 Date & Location Match"].push(result)
        } else if (dateMatch) {
          buckets["📅 Date Match Only"].push(result)
        } else if (locationMatch) {
          buckets["📍 Location Match Only"].push(result)
        } else {
          buckets["📄 Other Results"].push(result)
        }
      })
    } else if (hasDateFilter) {
      // Only date filter - create 2 buckets with ALL results
      buckets["📅 Date Match"] = []
      buckets["📄 Other Results"] = []
      
      results.forEach(result => {
        const dateMatch = matchesDate(result, dateFilter)
        if (dateMatch) {
          buckets["📅 Date Match"].push(result)
        } else {
          buckets["📄 Other Results"].push(result)
        }
      })
    } else if (hasLocationFilter) {
      // Only location filter - create 2 buckets with ALL results
      buckets["📍 Location Match"] = []
      buckets["📄 Other Results"] = []
      
      results.forEach(result => {
        const locationMatch = matchesLocation(result, locationFilter)
        if (locationMatch) {
          buckets["📍 Location Match"].push(result)
        } else {
          buckets["📄 Other Results"].push(result)
        }
      })
    }
    
    // Remove empty buckets and sort each bucket by similarity score
    Object.keys(buckets).forEach(key => {
      if (buckets[key].length === 0) {
        delete buckets[key]
      } else {
        // Sort each bucket by similarity score (highest to lowest)
        buckets[key].sort((a, b) => (b.similarity_score || 0) - (a.similarity_score || 0))
      }
    })
    
    return buckets
  }

  const getBucketColor = (bucketName: string) => {
    const colors = [
      '#3b82f6', // Blue
      '#10b981', // Green  
      '#f59e0b', // Orange
      '#ef4444', // Red
      '#8b5cf6', // Purple
      '#06b6d4', // Cyan
      '#84cc16', // Lime
    ]
    
    const hash = bucketName.split('').reduce((a, b) => {
      a = ((a << 5) - a) + b.charCodeAt(0)
      return a & a
    }, 0)
    
    return colors[Math.abs(hash) % colors.length]
  }

  // Enhanced thumbnail interactions
  const handleItemSelect = (filePath: string, event: React.MouseEvent) => {
    if (event.ctrlKey || event.metaKey) {
      // Multi-select with Ctrl/Cmd
      setSelectedItems(prev => {
        const newSet = new Set(prev)
        if (newSet.has(filePath)) {
          newSet.delete(filePath)
        } else {
          newSet.add(filePath)
        }
        return newSet
      })
    } else {
      // Single select
      setSelectedItems(new Set([filePath]))
    }
  }

  const handleItemOpen = async (filePath: string) => {
    if (window.electronAPI?.openFile) {
      try {
        await window.electronAPI.openFile(filePath)
        toast({
          title: "Opening File",
          description: `Opening ${filePath.split('/').pop()}`,
        })
      } catch (error) {
        toast({
          title: "Error",
          description: "Failed to open file",
          variant: "destructive",
        })
      }
    }
  }

  const handleItemReveal = async (filePath: string) => {
    if (window.electronAPI?.revealFile) {
      try {
        await window.electronAPI.revealFile(filePath)
        toast({
          title: "Revealing File",
          description: `Showing ${filePath.split('/').pop()} in finder`,
        })
      } catch (error) {
        toast({
          title: "Error", 
          description: "Failed to reveal file",
          variant: "destructive",
        })
      }
    }
  }

  const handleDragStart = (event: React.DragEvent, item: any) => {
    setDraggedItem(item)
    event.dataTransfer.setData("text/plain", item.file_path)
    event.dataTransfer.effectAllowed = "copy"
  }

  const handleDragEnd = () => {
    setDraggedItem(null)
  }

  // Render search results with buckets or simple grid
  const renderSearchResults = () => {
    // Sort results by similarity score (highest to lowest)
    const sortedResults = [...searchResults].sort((a, b) => (b.similarity_score || 0) - (a.similarity_score || 0))
    const buckets = organizeBuckets(sortedResults)
    const bucketEntries = Object.entries(buckets)
    
    // If only one bucket (no filters), show simple grid
    if (bucketEntries.length === 1 && bucketEntries[0][0] === "All Results") {
      return (
        <div className="grid grid-cols-2 sm:grid-cols-3 md:grid-cols-4 lg:grid-cols-5 xl:grid-cols-6 2xl:grid-cols-7 gap-4">
          {sortedResults.map((result: any, index: number) => (
            <SimpleCard
              key={index}
              result={result}
              isSelected={selectedItems.has(result.file_path)}
              onSelect={(event: React.MouseEvent) => handleItemSelect(result.file_path, event)}
              onOpen={() => handleItemOpen(result.file_path)}
              onReveal={() => handleItemReveal(result.file_path)}
            />
          ))}
        </div>
      )
    }
    
    // Multiple buckets - show with headers
    return (
      <div className="space-y-8">
        {bucketEntries.map(([bucketName, bucketResults]) => {
          const bucketColor = getBucketColor(bucketName)
          const isExpanded = expandedBuckets.has(bucketName)
          const maxVisible = 8
          const shouldShowToggle = bucketResults.length > maxVisible
          const visibleResults = isExpanded || !shouldShowToggle ? bucketResults : bucketResults.slice(0, maxVisible)
          
          return (
            <div key={bucketName} className="space-y-4">
              {/* Bucket header */}
              <div className="flex items-center gap-3">
                <div 
                  className="w-3 h-3 rounded-full flex-shrink-0" 
                  style={{ backgroundColor: bucketColor }}
                />
                <h3 className="text-lg font-semibold text-gray-900">{bucketName}</h3>
                <Badge variant="secondary" className="text-sm">
                  {bucketResults.length} {bucketResults.length === 1 ? 'item' : 'items'}
                </Badge>
                {shouldShowToggle && (
                  <Button
                    variant="ghost"
                    size="sm"
                    onClick={() => {
                      setExpandedBuckets(prev => {
                        const newSet = new Set(prev)
                        if (newSet.has(bucketName)) {
                          newSet.delete(bucketName)
                        } else {
                          newSet.add(bucketName)
                        }
                        return newSet
                      })
                    }}
                    className="ml-auto text-blue-600 hover:text-blue-700"
                  >
                    {isExpanded ? (
                      <>
                        <ChevronUp className="w-4 h-4 mr-1" />
                        Show Less
                      </>
                    ) : (
                      <>
                        <ChevronDown className="w-4 h-4 mr-1" />
                        Show {bucketResults.length - maxVisible} More
                      </>
                    )}
                  </Button>
                )}
              </div>
              
              {/* Results grid */}
              <div className="grid grid-cols-2 sm:grid-cols-3 md:grid-cols-4 lg:grid-cols-5 xl:grid-cols-6 2xl:grid-cols-7 gap-4">
                {visibleResults.map((result: any, index: number) => (
                  <SimpleCard
                    key={`${bucketName}-${index}`}
                    result={result}
                    isSelected={selectedItems.has(result.file_path)}
                    onSelect={(event: React.MouseEvent) => handleItemSelect(result.file_path, event)}
                    onOpen={() => handleItemOpen(result.file_path)}
                    onReveal={() => handleItemReveal(result.file_path)}
                  />
                ))}
              </div>
            </div>
          )
        })}
      </div>
    )
  }

  const handleSearch = async () => {
    if (isSearchInProgress) {
      return
    }
    
    if (searchQuery.trim() && window.electronAPI) {
      setIsSearchInProgress(true)
      setIsSearching(true)
      setHasSearched(false)
      setSearchResults([])
      
      // Add timeout to prevent infinite searching state
      const timeout = setTimeout(() => {
        setIsSearching(false)
        setHasSearched(true)
        setSearchTimeout(null)
        setIsSearchInProgress(false)
        toast({
          title: "Search Timeout",
          description: "Search took too long. Please try again.",
          variant: "destructive",
        })
      }, 30000) // 30 second timeout
      setSearchTimeout(timeout)

      try {
        // Pass content type filter and manual filters to search
        const contentTypes = selectedContentType === "all" ? undefined : [selectedContentType]
        
        const result = await window.electronAPI.searchVideos(searchQuery, { 
          content_types: contentTypes,
          date_filter: dateFilter.trim(),
          location_filter: locationFilter.trim()
        })
        
        const filterDescription = [
          searchQuery,
          selectedContentType !== "all" ? `(${selectedContentType})` : "",
          dateFilter.trim() ? `Date: ${dateFilter}` : "",
          locationFilter.trim() ? `Location: ${locationFilter}` : ""
        ].filter(Boolean).join(" ")
        
      toast({
        title: "Search Started",
          description: `Searching for: ${filterDescription}`,
        })
      } catch (error) {
        clearTimeout(timeout)
        setSearchTimeout(null)
        setIsSearching(false)
        setHasSearched(true)
        setIsSearchInProgress(false)
        toast({
          title: "Search Error",
          description: "Failed to start search",
          variant: "destructive",
      })
      }
    }
  }

  return (
    <div className="flex-1 space-y-8 p-8">
      {/* Header with prominent upload section */}
      <div className="space-y-6">
        <div>
          <h1 className="text-4xl font-bold text-gray-900 mb-2">Dashboard</h1>
          <p className="text-lg text-gray-600">Search and organize your video content with AI-powered tagging</p>
        </div>


      </div>

      {/* Stats Cards */}
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6">
        <Card className="border-0 shadow-sm bg-white">
          <CardContent className="p-6">
            <div className="flex items-center justify-between">
              <div>
                <p className="text-sm font-medium text-gray-600">Total Videos</p>
                <p className="text-3xl font-bold text-gray-900 mt-1">{stats.totalVideos.toLocaleString()}</p>
                <p className="text-sm text-blue-600 mt-1">
                  {stats.searchServerStatus ? "Server Online" : "Server Offline"}
                </p>
              </div>
              <div className="p-4 bg-blue-50 rounded-xl">
                <Video className="w-7 h-7 text-blue-600" />
              </div>
            </div>
          </CardContent>
        </Card>
        
        <Card className="border-0 shadow-sm bg-white">
          <CardContent className="p-6">
            <div className="flex items-center justify-between">
              <div>
                <p className="text-sm font-medium text-gray-600">Total Images</p>
                <p className="text-3xl font-bold text-gray-900 mt-1">{stats.totalImages.toLocaleString()}</p>
                <p className="text-sm text-gray-600 mt-1">Ready to search</p>
              </div>
              <div className="p-4 bg-green-50 rounded-xl">
                <FileText className="w-7 h-7 text-green-600" />
              </div>
            </div>
          </CardContent>
        </Card>
        
        <Card className="border-0 shadow-sm bg-white">
          <CardContent className="p-6">
            <div className="flex items-center justify-between">
              <div>
                <p className="text-sm font-medium text-gray-600">Text Documents</p>
                <p className="text-3xl font-bold text-gray-900 mt-1">{stats.totalText.toLocaleString()}</p>
                <p className="text-sm text-gray-600 mt-1">Analyzed</p>
              </div>
              <div className="p-4 bg-purple-50 rounded-xl">
                <FileText className="w-7 h-7 text-purple-600" />
              </div>
            </div>
          </CardContent>
        </Card>

        <Card className="border-0 shadow-sm bg-white">
            <CardContent className="p-6">
              <div className="flex items-center justify-between">
                <div>
                <p className="text-sm font-medium text-gray-600">Audio Files</p>
                <p className="text-3xl font-bold text-gray-900 mt-1">{stats.totalAudio.toLocaleString()}</p>
                <p className="text-sm text-gray-600 mt-1">Processed</p>
                </div>
              <div className="p-4 bg-orange-50 rounded-xl">
                <FileText className="w-7 h-7 text-orange-600" />
                </div>
              </div>
            </CardContent>
          </Card>
      </div>

      {/* Search and Filters */}
      <Card className="border-0 shadow-sm bg-white">
        <CardContent className="p-6">
          <div className="space-y-4">
          <div className="flex flex-col lg:flex-row gap-4">
            <div className="flex-1">
              <div className="relative">
                <Search className="absolute left-4 top-1/2 transform -translate-y-1/2 text-gray-400 w-5 h-5" />
                <Input
                  placeholder="Search videos by content, mood, or context..."
                  value={searchQuery}
                  onChange={(e) => setSearchQuery(e.target.value)}
                  onKeyPress={(e) => e.key === "Enter" && handleSearch()}
                  className="pl-12 h-14 text-base border-gray-200 focus:border-blue-500 focus:ring-blue-500"
                />
              </div>
            </div>
            <div className="flex gap-3">
                <Select value={selectedContentType} onValueChange={setSelectedContentType}>
                <SelectTrigger className="w-36 h-14">
                  <SelectValue placeholder="Type" />
                </SelectTrigger>
                <SelectContent>
                  <SelectItem value="all">All Types</SelectItem>
                  <SelectItem value="video">Video</SelectItem>
                    <SelectItem value="image">Image</SelectItem>
                    <SelectItem value="text">Text</SelectItem>
                  <SelectItem value="audio">Audio</SelectItem>
                </SelectContent>
              </Select>
                <Button
                  variant="outline"
                  size="lg"
                  className="h-14 px-4"
                  onClick={() => setShowAdvancedFilters(!showAdvancedFilters)}
                >
                <Filter className="w-4 h-4 mr-2" />
                Filters
              </Button>
              <Button size="lg" className="h-14 px-8 bg-blue-600 hover:bg-blue-700" onClick={handleSearch}>
                Search
              </Button>
            </div>
            </div>

            {/* Advanced Filters */}
            {showAdvancedFilters && (
              <div className="border-t pt-4 space-y-4">
                <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                  <div className="space-y-2">
                    <label className="text-sm font-medium text-gray-700 flex items-center">
                      <Calendar className="w-4 h-4 mr-2" />
                      Date Filter
                    </label>
                    <Input
                      placeholder="e.g., 2023-12-25 or 2023-12"
                      value={dateFilter}
                      onChange={(e) => setDateFilter(e.target.value)}
                      className="h-10"
                    />
                    <p className="text-xs text-gray-500">Search for content from specific dates</p>
                  </div>
                  <div className="space-y-2">
                    <label className="text-sm font-medium text-gray-700 flex items-center">
                      <MapPin className="w-4 h-4 mr-2" />
                      Location Filter
                    </label>
                    <Input
                      placeholder="e.g., San Francisco, beach, etc."
                      value={locationFilter}
                      onChange={(e) => setLocationFilter(e.target.value)}
                      className="h-10"
                    />
                    <p className="text-xs text-gray-500">Search for content from specific locations</p>
                  </div>
                </div>
                
                {(dateFilter || locationFilter) && (
                  <div className="flex items-center gap-2">
                    <Badge variant="secondary" className="text-sm">
                      Filters will create organized buckets in results
                    </Badge>
                    <Button
                      variant="ghost"
                      size="sm"
                      onClick={() => {
                        setDateFilter("")
                        setLocationFilter("")
                      }}
                      className="text-xs"
                    >
                      Clear Filters
                    </Button>
                  </div>
                )}
              </div>
            )}
          </div>
        </CardContent>
      </Card>

      {/* Results Section */}
      <Card className="border-0 shadow-sm bg-white">
        <CardHeader className="pb-4">
          <div className="flex items-center justify-between">
            <div>
              <CardTitle className="text-xl text-gray-900">Search Results</CardTitle>
              <CardDescription>
                {isSearching ? "Searching..." : 
                 hasSearched ? `Found ${searchResults.length} results` : 
                 "Enter a search query to find content"}
              </CardDescription>
            </div>
            <div className="flex items-center gap-2">
              <Button
                variant={viewMode === "grid" ? "default" : "outline"}
                size="sm"
                onClick={() => setViewMode("grid")}
                className={viewMode === "grid" ? "bg-blue-600 hover:bg-blue-700" : ""}
              >
                <Grid className="w-4 h-4" />
              </Button>
              <Button
                variant={viewMode === "list" ? "default" : "outline"}
                size="sm"
                onClick={() => setViewMode("list")}
                className={viewMode === "list" ? "bg-blue-600 hover:bg-blue-700" : ""}
              >
                <List className="w-4 h-4" />
              </Button>
            </div>
          </div>
        </CardHeader>
        <CardContent className="max-w-full overflow-hidden">
          {isSearching ? (
            <div className="flex flex-col items-center justify-center py-12 text-center">
              <div className="w-8 h-8 border-4 border-blue-600 border-t-transparent rounded-full animate-spin mb-4"></div>
              <h3 className="text-lg font-semibold text-gray-900 mb-2">Searching...</h3>
              <p className="text-gray-500">Finding content matching your query</p>
                    </div>
          ) : !hasSearched ? (
            <div className="flex flex-col items-center justify-center py-12 text-center">
              <Search className="w-12 h-12 text-gray-400 mb-4" />
              <h3 className="text-lg font-semibold text-gray-900 mb-2">Start searching</h3>
              <p className="text-gray-500 max-w-md">Enter a query above to search through your content</p>
                  </div>
          ) : searchResults.length === 0 ? (
            <div className="flex flex-col items-center justify-center py-12 text-center">
              <Search className="w-12 h-12 text-gray-400 mb-4" />
              <h3 className="text-lg font-semibold text-gray-900 mb-2">No results found</h3>
              <p className="text-gray-500 max-w-md">Try different keywords or check your spelling</p>
            </div>
          ) : (
            <div className="w-full">
              {renderSearchResults()}
              
              {/* Selection info */}
              {selectedItems.size > 0 && (
                <div className="fixed bottom-6 left-1/2 transform -translate-x-1/2 bg-blue-600 text-white px-4 py-2 rounded-lg shadow-lg flex items-center gap-3 z-50">
                  <span>{selectedItems.size} item{selectedItems.size !== 1 ? 's' : ''} selected</span>
                  <button
                    onClick={() => setSelectedItems(new Set())}
                    className="text-blue-200 hover:text-white"
                  >
                    Clear
                  </button>
                        </div>
              )}
            </div>
          )}
        </CardContent>
      </Card>
    </div>
  )
}
