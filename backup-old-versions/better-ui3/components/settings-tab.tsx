"use client"

import { Folder, Cpu, Database, Bell } from "lucide-react"
import { Button } from "@/components/ui/button"
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card"
import { Input } from "@/components/ui/input"
import { Label } from "@/components/ui/label"
import { Switch } from "@/components/ui/switch"
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select"

export function SettingsTab() {
  return (
    <div className="flex-1 space-y-8 p-8">
      {/* Header */}
      <div>
        <h1 className="text-3xl font-bold text-gray-900 mb-2">Settings</h1>
        <p className="text-gray-600">Configure your application preferences</p>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-2 gap-8">
        {/* Storage Settings */}
        <Card className="border-0 shadow-sm bg-white">
          <CardHeader>
            <CardTitle className="flex items-center gap-2">
              <Folder className="w-5 h-5" />
              Storage & Paths
            </CardTitle>
            <CardDescription>Configure where your files are stored and processed</CardDescription>
          </CardHeader>
          <CardContent className="space-y-4">
            <div className="space-y-2">
              <Label htmlFor="video-path">Default Video Directory</Label>
              <div className="flex gap-2">
                <Input id="video-path" placeholder="/Users/username/Videos" className="flex-1" />
                <Button variant="outline">Browse</Button>
              </div>
            </div>
            <div className="space-y-2">
              <Label htmlFor="cache-path">Cache Directory</Label>
              <div className="flex gap-2">
                <Input id="cache-path" placeholder="/Users/username/.videosearch/cache" className="flex-1" />
                <Button variant="outline">Browse</Button>
              </div>
            </div>
            <div className="flex items-center justify-between">
              <div className="space-y-0.5">
                <Label>Auto-organize files</Label>
                <p className="text-sm text-gray-500">Automatically organize imported files by date</p>
              </div>
              <Switch />
            </div>
          </CardContent>
        </Card>

        {/* Processing Settings */}
        <Card className="border-0 shadow-sm bg-white">
          <CardHeader>
            <CardTitle className="flex items-center gap-2">
              <Cpu className="w-5 h-5" />
              Processing
            </CardTitle>
            <CardDescription>AI processing and performance settings</CardDescription>
          </CardHeader>
          <CardContent className="space-y-4">
            <div className="space-y-2">
              <Label htmlFor="cpu-threads">CPU Threads</Label>
              <Select defaultValue="auto">
                <SelectTrigger>
                  <SelectValue />
                </SelectTrigger>
                <SelectContent>
                  <SelectItem value="auto">Auto (Recommended)</SelectItem>
                  <SelectItem value="1">1 Thread</SelectItem>
                  <SelectItem value="2">2 Threads</SelectItem>
                  <SelectItem value="4">4 Threads</SelectItem>
                  <SelectItem value="8">8 Threads</SelectItem>
                </SelectContent>
              </Select>
            </div>
            <div className="space-y-2">
              <Label htmlFor="quality">Processing Quality</Label>
              <Select defaultValue="high">
                <SelectTrigger>
                  <SelectValue />
                </SelectTrigger>
                <SelectContent>
                  <SelectItem value="fast">Fast (Lower accuracy)</SelectItem>
                  <SelectItem value="balanced">Balanced</SelectItem>
                  <SelectItem value="high">High Quality (Slower)</SelectItem>
                </SelectContent>
              </Select>
            </div>
            <div className="flex items-center justify-between">
              <div className="space-y-0.5">
                <Label>GPU Acceleration</Label>
                <p className="text-sm text-gray-500">Use GPU for faster processing (if available)</p>
              </div>
              <Switch />
            </div>
          </CardContent>
        </Card>

        {/* Database Settings */}
        <Card className="border-0 shadow-sm bg-white">
          <CardHeader>
            <CardTitle className="flex items-center gap-2">
              <Database className="w-5 h-5" />
              Database
            </CardTitle>
            <CardDescription>Manage your video database and indexing</CardDescription>
          </CardHeader>
          <CardContent className="space-y-4">
            <div className="flex items-center justify-between">
              <div className="space-y-0.5">
                <Label>Auto-backup database</Label>
                <p className="text-sm text-gray-500">Automatically backup your video index daily</p>
              </div>
              <Switch defaultChecked />
            </div>
            <div className="flex items-center justify-between">
              <div className="space-y-0.5">
                <Label>Index new files automatically</Label>
                <p className="text-sm text-gray-500">Process new files as they're added</p>
              </div>
              <Switch defaultChecked />
            </div>
            <div className="pt-4 space-y-2">
              <Button variant="outline" className="w-full">
                Rebuild Search Index
              </Button>
              <Button variant="outline" className="w-full">
                Export Database
              </Button>
            </div>
          </CardContent>
        </Card>

        {/* Notifications */}
        <Card className="border-0 shadow-sm bg-white">
          <CardHeader>
            <CardTitle className="flex items-center gap-2">
              <Bell className="w-5 h-5" />
              Notifications
            </CardTitle>
            <CardDescription>Configure when to receive notifications</CardDescription>
          </CardHeader>
          <CardContent className="space-y-4">
            <div className="flex items-center justify-between">
              <div className="space-y-0.5">
                <Label>Processing complete</Label>
                <p className="text-sm text-gray-500">Notify when video processing finishes</p>
              </div>
              <Switch defaultChecked />
            </div>
            <div className="flex items-center justify-between">
              <div className="space-y-0.5">
                <Label>Errors and warnings</Label>
                <p className="text-sm text-gray-500">Show notifications for processing errors</p>
              </div>
              <Switch defaultChecked />
            </div>
            <div className="flex items-center justify-between">
              <div className="space-y-0.5">
                <Label>Storage warnings</Label>
                <p className="text-sm text-gray-500">Alert when storage space is low</p>
              </div>
              <Switch defaultChecked />
            </div>
          </CardContent>
        </Card>
      </div>
    </div>
  )
}
