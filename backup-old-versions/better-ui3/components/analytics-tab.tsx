"use client"

import { BarChart3, TrendingUp, Clock, FileVideo } from "lucide-react"
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card"

export function AnalyticsTab() {
  const analyticsData = [
    { label: "Videos Processed", value: "1,247", change: "+12%", icon: FileVideo },
    { label: "Total Processing Time", value: "156 hrs", change: "+8%", icon: Clock },
    { label: "Average File Size", value: "485 MB", change: "-3%", icon: BarChart3 },
    { label: "Search Accuracy", value: "94.2%", change: "+2%", icon: TrendingUp },
  ]

  return (
    <div className="flex-1 space-y-8 p-8">
      {/* Header */}
      <div>
        <h1 className="text-3xl font-bold text-gray-900 mb-2">Analytics</h1>
        <p className="text-gray-600">Performance insights and usage statistics</p>
      </div>

      {/* Stats Grid */}
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6">
        {analyticsData.map((stat, index) => (
          <Card key={index} className="border-0 shadow-sm bg-white">
            <CardContent className="p-6">
              <div className="flex items-center justify-between">
                <div>
                  <p className="text-sm font-medium text-gray-600">{stat.label}</p>
                  <p className="text-2xl font-bold text-gray-900 mt-1">{stat.value}</p>
                  <p className={`text-sm mt-1 ${stat.change.startsWith("+") ? "text-green-600" : "text-red-600"}`}>
                    {stat.change} from last month
                  </p>
                </div>
                <div className="p-3 bg-blue-50 rounded-lg">
                  <stat.icon className="w-6 h-6 text-blue-600" />
                </div>
              </div>
            </CardContent>
          </Card>
        ))}
      </div>

      {/* Charts Placeholder */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        <Card className="border-0 shadow-sm bg-white">
          <CardHeader>
            <CardTitle>Processing Timeline</CardTitle>
            <CardDescription>Video processing over time</CardDescription>
          </CardHeader>
          <CardContent>
            <div className="h-64 flex items-center justify-center text-gray-400">
              <div className="text-center">
                <BarChart3 className="w-12 h-12 mx-auto mb-4" />
                <p>Chart visualization will be integrated with Python analytics</p>
              </div>
            </div>
          </CardContent>
        </Card>

        <Card className="border-0 shadow-sm bg-white">
          <CardHeader>
            <CardTitle>Content Categories</CardTitle>
            <CardDescription>Distribution of video content types</CardDescription>
          </CardHeader>
          <CardContent>
            <div className="h-64 flex items-center justify-center text-gray-400">
              <div className="text-center">
                <TrendingUp className="w-12 h-12 mx-auto mb-4" />
                <p>Category breakdown charts coming soon</p>
              </div>
            </div>
          </CardContent>
        </Card>
      </div>
    </div>
  )
}
