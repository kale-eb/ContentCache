"use client"

import { useState } from "react"
import { Search, Video, FileText, Users, Upload, Filter, Grid, List } from "lucide-react"
import { Button } from "@/components/ui/button"
import { Input } from "@/components/ui/input"
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card"
import { Badge } from "@/components/ui/badge"
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select"

export function PremiumDashboard() {
  const [searchQuery, setSearchQuery] = useState("")
  const [viewMode, setViewMode] = useState<"grid" | "list">("grid")

  const videoResults = [
    {
      id: 1,
      title: "Product Demo - Q4 2024",
      duration: "12:34",
      thumbnail: "/placeholder.svg?height=120&width=200",
      tags: ["demo", "product", "presentation"],
      uploadDate: "2024-01-15",
      size: "245 MB",
    },
    {
      id: 2,
      title: "Team Meeting - Strategy Discussion",
      duration: "45:12",
      thumbnail: "/placeholder.svg?height=120&width=200",
      tags: ["meeting", "strategy", "team"],
      uploadDate: "2024-01-14",
      size: "1.2 GB",
    },
    {
      id: 3,
      title: "Customer Interview - User Feedback",
      duration: "28:45",
      thumbnail: "/placeholder.svg?height=120&width=200",
      tags: ["interview", "customer", "feedback"],
      uploadDate: "2024-01-13",
      size: "567 MB",
    },
    {
      id: 4,
      title: "Training Session - New Features",
      duration: "35:20",
      thumbnail: "/placeholder.svg?height=120&width=200",
      tags: ["training", "features", "education"],
      uploadDate: "2024-01-12",
      size: "890 MB",
    },
  ]

  const stats = [
    { label: "Total Videos", value: "1,247", icon: Video, change: "+12%" },
    { label: "Storage Used", value: "2.4 TB", icon: FileText, change: "+8%" },
    { label: "Monthly Searches", value: "15.2K", icon: Search, change: "+23%" },
    { label: "Active Users", value: "342", icon: Users, change: "+5%" },
  ]

  return (
    <div className="flex-1 space-y-6 p-6">
      {/* Header */}
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-3xl font-bold text-gray-900">Dashboard</h1>
          <p className="text-gray-600 mt-1">Search and organize your video content with AI-powered tagging</p>
        </div>
        <Button className="bg-blue-600 hover:bg-blue-700">
          <Upload className="w-4 h-4 mr-2" />
          Upload Video
        </Button>
      </div>

      {/* Stats Cards */}
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6">
        {stats.map((stat, index) => (
          <Card key={index} className="border-0 shadow-sm bg-white">
            <CardContent className="p-6">
              <div className="flex items-center justify-between">
                <div>
                  <p className="text-sm font-medium text-gray-600">{stat.label}</p>
                  <p className="text-2xl font-bold text-gray-900 mt-1">{stat.value}</p>
                  <p className="text-sm text-green-600 mt-1">{stat.change} from last month</p>
                </div>
                <div className="p-3 bg-blue-50 rounded-lg">
                  <stat.icon className="w-6 h-6 text-blue-600" />
                </div>
              </div>
            </CardContent>
          </Card>
        ))}
      </div>

      {/* Search and Filters */}
      <Card className="border-0 shadow-sm bg-white">
        <CardContent className="p-6">
          <div className="flex flex-col lg:flex-row gap-4">
            <div className="flex-1">
              <div className="relative">
                <Search className="absolute left-3 top-1/2 transform -translate-y-1/2 text-gray-400 w-4 h-4" />
                <Input
                  placeholder="Search videos by content, mood, or context..."
                  value={searchQuery}
                  onChange={(e) => setSearchQuery(e.target.value)}
                  className="pl-10 h-12 border-gray-200 focus:border-blue-500 focus:ring-blue-500"
                />
              </div>
            </div>
            <div className="flex gap-3">
              <Select defaultValue="all">
                <SelectTrigger className="w-32 h-12">
                  <SelectValue placeholder="Type" />
                </SelectTrigger>
                <SelectContent>
                  <SelectItem value="all">All Types</SelectItem>
                  <SelectItem value="video">Video</SelectItem>
                  <SelectItem value="audio">Audio</SelectItem>
                </SelectContent>
              </Select>
              <Button variant="outline" size="lg" className="h-12">
                <Filter className="w-4 h-4 mr-2" />
                Filters
              </Button>
              <Button size="lg" className="h-12 bg-blue-600 hover:bg-blue-700">
                Search
              </Button>
            </div>
          </div>
        </CardContent>
      </Card>

      {/* Results Section */}
      <Card className="border-0 shadow-sm bg-white">
        <CardHeader className="pb-4">
          <div className="flex items-center justify-between">
            <div>
              <CardTitle className="text-xl text-gray-900">Search Results</CardTitle>
              <CardDescription>Found {videoResults.length} videos matching your criteria</CardDescription>
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
        <CardContent>
          {viewMode === "grid" ? (
            <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 xl:grid-cols-4 gap-6">
              {videoResults.map((video) => (
                <Card
                  key={video.id}
                  className="border border-gray-200 hover:shadow-md transition-shadow cursor-pointer"
                >
                  <div className="relative">
                    <img
                      src={video.thumbnail || "/placeholder.svg"}
                      alt={video.title}
                      className="w-full h-32 object-cover rounded-t-lg"
                    />
                    <div className="absolute bottom-2 right-2 bg-black bg-opacity-75 text-white text-xs px-2 py-1 rounded">
                      {video.duration}
                    </div>
                  </div>
                  <CardContent className="p-4">
                    <h3 className="font-semibold text-gray-900 mb-2 line-clamp-2">{video.title}</h3>
                    <div className="flex flex-wrap gap-1 mb-3">
                      {video.tags.map((tag) => (
                        <Badge key={tag} variant="secondary" className="text-xs bg-blue-50 text-blue-700">
                          {tag}
                        </Badge>
                      ))}
                    </div>
                    <div className="text-sm text-gray-500 space-y-1">
                      <p>Uploaded: {video.uploadDate}</p>
                      <p>Size: {video.size}</p>
                    </div>
                  </CardContent>
                </Card>
              ))}
            </div>
          ) : (
            <div className="space-y-4">
              {videoResults.map((video) => (
                <Card
                  key={video.id}
                  className="border border-gray-200 hover:shadow-md transition-shadow cursor-pointer"
                >
                  <CardContent className="p-4">
                    <div className="flex items-center gap-4">
                      <img
                        src={video.thumbnail || "/placeholder.svg"}
                        alt={video.title}
                        className="w-24 h-16 object-cover rounded"
                      />
                      <div className="flex-1">
                        <h3 className="font-semibold text-gray-900 mb-1">{video.title}</h3>
                        <div className="flex flex-wrap gap-1 mb-2">
                          {video.tags.map((tag) => (
                            <Badge key={tag} variant="secondary" className="text-xs bg-blue-50 text-blue-700">
                              {tag}
                            </Badge>
                          ))}
                        </div>
                        <div className="flex items-center gap-4 text-sm text-gray-500">
                          <span>Duration: {video.duration}</span>
                          <span>Size: {video.size}</span>
                          <span>Uploaded: {video.uploadDate}</span>
                        </div>
                      </div>
                    </div>
                  </CardContent>
                </Card>
              ))}
            </div>
          )}
        </CardContent>
      </Card>
    </div>
  )
}
