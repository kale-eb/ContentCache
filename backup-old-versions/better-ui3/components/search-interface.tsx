import { Search, Filter, Upload, Zap, BarChart3 } from "lucide-react"
import { Button } from "@/components/ui/button"
import { Input } from "@/components/ui/input"
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card"
import { Badge } from "@/components/ui/badge"
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select"
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs"

export function SearchInterface() {
  return (
    <div className="space-y-6">
      {/* Search Header */}
      <div className="space-y-2">
        <h1 className="text-3xl font-bold tracking-tight text-gray-900">Content Search</h1>
        <p className="text-gray-600">Search and organize your video content with AI-powered tagging</p>
      </div>

      {/* Quick Actions */}
      <div className="flex flex-wrap gap-3">
        <Button className="bg-blue-600 hover:bg-blue-700">
          <Upload className="mr-2 h-4 w-4" />
          Import Videos
        </Button>
        <Button variant="outline">
          <Zap className="mr-2 h-4 w-4" />
          Batch Process
        </Button>
        <Button variant="outline">
          <BarChart3 className="mr-2 h-4 w-4" />
          Analytics
        </Button>
      </div>

      {/* Search Interface */}
      <Card className="border-0 shadow-sm">
        <CardHeader className="pb-4">
          <CardTitle className="text-lg">Search Your Library</CardTitle>
          <CardDescription>Use natural language to find videos by content, mood, or context</CardDescription>
        </CardHeader>
        <CardContent className="space-y-4">
          <div className="flex gap-2">
            <div className="relative flex-1">
              <Search className="absolute left-3 top-1/2 h-4 w-4 -translate-y-1/2 text-gray-400" />
              <Input
                placeholder="Search videos by content, mood, or context..."
                className="pl-10 h-12 text-base border-gray-200 focus:border-blue-500 focus:ring-blue-500"
              />
            </div>
            <Select defaultValue="video">
              <SelectTrigger className="w-32 h-12">
                <SelectValue />
              </SelectTrigger>
              <SelectContent>
                <SelectItem value="video">Video</SelectItem>
                <SelectItem value="audio">Audio</SelectItem>
                <SelectItem value="text">Text</SelectItem>
                <SelectItem value="all">All</SelectItem>
              </SelectContent>
            </Select>
            <Button size="lg" className="px-8 bg-blue-600 hover:bg-blue-700">
              Search
            </Button>
          </div>

          <div className="flex items-center gap-4 text-sm text-gray-600">
            <Button variant="ghost" size="sm">
              <Filter className="mr-2 h-4 w-4" />
              Advanced Filters
            </Button>
            <div className="flex gap-2">
              <Badge variant="secondary">Recent: "product demo"</Badge>
              <Badge variant="secondary">Recent: "team meeting"</Badge>
            </div>
          </div>
        </CardContent>
      </Card>

      {/* Search Modes */}
      <Card className="border-0 shadow-sm">
        <CardHeader className="pb-4">
          <CardTitle className="text-lg">Search Configuration</CardTitle>
        </CardHeader>
        <CardContent>
          <Tabs defaultValue="fast" className="w-full">
            <TabsList className="grid w-full grid-cols-2">
              <TabsTrigger value="fast">Fast Search</TabsTrigger>
              <TabsTrigger value="server">Search Server</TabsTrigger>
            </TabsList>
            <TabsContent value="fast" className="mt-4">
              <div className="space-y-2">
                <h4 className="font-medium">Fast Search Mode</h4>
                <p className="text-sm text-gray-600">Command-line search (~13s per search, auto-builds cache)</p>
                <div className="flex items-center gap-2 mt-3">
                  <Badge className="bg-green-100 text-green-700">Recommended</Badge>
                  <Badge variant="outline">Local Processing</Badge>
                </div>
              </div>
            </TabsContent>
            <TabsContent value="server" className="mt-4">
              <div className="space-y-2">
                <h4 className="font-medium">Search Server Mode</h4>
                <p className="text-sm text-gray-600">HTTP API (~1s per search after startup)</p>
                <div className="flex items-center gap-2 mt-3">
                  <Badge className="bg-blue-100 text-blue-700">Enterprise</Badge>
                  <Badge variant="outline">API Access</Badge>
                </div>
              </div>
            </TabsContent>
          </Tabs>
        </CardContent>
      </Card>

      {/* Results Placeholder */}
      <Card className="border-0 shadow-sm">
        <CardHeader>
          <CardTitle className="text-lg">Search Results</CardTitle>
          <CardDescription>Your search results will appear here</CardDescription>
        </CardHeader>
        <CardContent>
          <div className="flex items-center justify-center h-32 text-gray-400">
            <div className="text-center">
              <Search className="h-8 w-8 mx-auto mb-2" />
              <p>Start searching to see results</p>
            </div>
          </div>
        </CardContent>
      </Card>
    </div>
  )
}
