import React from 'react'
import { Card, CardHeader, CardTitle, CardContent } from './ui'
import { ChartContainer, ChartTooltip, ChartTooltipContent } from './ui/chart'
import { BarChart, Bar, XAxis, YAxis, ResponsiveContainer, Cell } from 'recharts'
import { Icon } from './ui'

interface FeatureImportanceChartProps {
  inferenceData: any[]
  currentTime: number
  fps?: number
}

export const FeatureImportanceChart: React.FC<FeatureImportanceChartProps> = ({
  inferenceData,
  currentTime,
  fps = 30
}) => {
  // Find the most relevant inference for current time
  const getCurrentInference = () => {
    if (!inferenceData || inferenceData.length === 0) return null

    const currentFrame = Math.floor(currentTime * fps)

    // Find the inference result that covers the current frame
    for (let i = 0; i < inferenceData.length; i++) {
      const current = inferenceData[i]
      const next = inferenceData[i + 1]

      const startFrame = current.frame
      const endFrame = next ? next.frame - 1 : Infinity

      if (currentFrame >= startFrame && currentFrame <= endFrame) {
        return current
      }
    }

    // If no exact match, return the last inference
    return inferenceData[inferenceData.length - 1]
  }

  const currentInference = getCurrentInference()

  if (!currentInference || !currentInference.feature_importance) {
    return (
      <Card className="shadow-lg border-0 bg-white/80 backdrop-blur-sm">
        <CardHeader className="pb-4">
          <CardTitle className="flex items-center gap-2 text-xl">
            <Icon name="BarChart3" className="h-5 w-5 text-purple-600" />
            Feature Importance
          </CardTitle>
        </CardHeader>
        <CardContent>
          <div className="p-8 text-center text-slate-500">
            <Icon name="BarChart3" className="h-12 w-12 mx-auto mb-4 opacity-50" />
            <p className="text-sm">Feature importance data will appear here during playback</p>
          </div>
        </CardContent>
      </Card>
    )
  }

  const { feature_importance, frame, class: emotionClass } = currentInference

  // Prepare data for charts
  const videoData = feature_importance.video.slice(0, 8).map((feature: any, index: number) => ({
    dimension: `Dim ${feature.dimension}`,
    importance: Math.abs(feature.importance),
    originalImportance: feature.importance,
    rank: index + 1
  }))

  const audioData = feature_importance.audio.slice(0, 8).map((feature: any, index: number) => ({
    dimension: `Dim ${feature.dimension}`,
    importance: Math.abs(feature.importance),
    originalImportance: feature.importance,
    rank: index + 1
  }))

  const chartConfig = {
    importance: {
      label: "Importance",
    },
  }

  return (
    <Card className="shadow-lg border-0 bg-white/80 backdrop-blur-sm">
      <CardHeader className="pb-4">
        <CardTitle className="flex items-center gap-2 text-xl">
          <Icon name="BarChart3" className="h-5 w-5 text-purple-600" />
          Feature Importance
        </CardTitle>
        <div className="text-sm text-slate-600">
          Frame {frame} - Emotion: <span className="font-semibold text-purple-600">{emotionClass}</span>
        </div>
      </CardHeader>
      <CardContent>
        <div className="grid gap-6 md:grid-cols-2">
          {/* Video Features Chart */}
          <div>
            <h3 className="text-sm font-medium text-slate-700 mb-4 flex items-center gap-2">
              <Icon name="Video" className="h-4 w-4 text-blue-600" />
              Top Video Features
            </h3>
            <ChartContainer config={chartConfig} className="h-[200px] w-full">
              <BarChart data={videoData} margin={{ top: 5, right: 5, left: 5, bottom: 5 }}>
                <XAxis
                  dataKey="dimension"
                  axisLine={false}
                  tickLine={false}
                  tick={{ fontSize: 10 }}
                  angle={-45}
                  textAnchor="end"
                  height={60}
                />
                <YAxis hide />
                <ChartTooltip
                  content={<ChartTooltipContent
                    formatter={(value, name, props) => [
                      `${props.payload.originalImportance >= 0 ? '+' : ''}${props.payload.originalImportance.toFixed(4)}`,
                      "Importance"
                    ]}
                  />}
                />
                <Bar dataKey="importance" radius={[2, 2, 0, 0]}>
                  {videoData.map((entry: any, index: number) => (
                    <Cell
                      key={`cell-${index}`}
                      fill={entry.originalImportance >= 0 ? "#3b82f6" : "#ef4444"}
                    />
                  ))}
                </Bar>
              </BarChart>
            </ChartContainer>
          </div>

          {/* Audio Features Chart */}
          <div>
            <h3 className="text-sm font-medium text-slate-700 mb-4 flex items-center gap-2">
              <Icon name="Volume2" className="h-4 w-4 text-green-600" />
              Top Audio Features
            </h3>
            <ChartContainer config={chartConfig} className="h-[200px] w-full">
              <BarChart data={audioData} margin={{ top: 5, right: 5, left: 5, bottom: 5 }}>
                <XAxis
                  dataKey="dimension"
                  axisLine={false}
                  tickLine={false}
                  tick={{ fontSize: 10 }}
                  angle={-45}
                  textAnchor="end"
                  height={60}
                />
                <YAxis hide />
                <ChartTooltip
                  content={<ChartTooltipContent
                    formatter={(value, name, props) => [
                      `${props.payload.originalImportance >= 0 ? '+' : ''}${props.payload.originalImportance.toFixed(4)}`,
                      "Importance"
                    ]}
                  />}
                />
                <Bar dataKey="importance" radius={[2, 2, 0, 0]}>
                  {audioData.map((entry: any, index: number) => (
                    <Cell
                      key={`cell-${index}`}
                      fill={entry.originalImportance >= 0 ? "#10b981" : "#ef4444"}
                    />
                  ))}
                </Bar>
              </BarChart>
            </ChartContainer>
          </div>
        </div>

        {/* Legend */}
        <div className="mt-4 flex items-center justify-center gap-6 text-xs text-slate-500">
          <div className="flex items-center gap-1">
            <div className="w-3 h-3 bg-blue-500 rounded"></div>
            <span>Positive Video</span>
          </div>
          <div className="flex items-center gap-1">
            <div className="w-3 h-3 bg-red-500 rounded"></div>
            <span>Negative Video</span>
          </div>
          <div className="flex items-center gap-1">
            <div className="w-3 h-3 bg-green-500 rounded"></div>
            <span>Positive Audio</span>
          </div>
          <div className="flex items-center gap-1">
            <div className="w-3 h-3 bg-red-500 rounded"></div>
            <span>Negative Audio</span>
          </div>
        </div>
      </CardContent>
    </Card>
  )
}