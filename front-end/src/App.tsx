import React, { useState } from 'react'
import { Button, Input, Label, Icon, Card, CardHeader, CardTitle, CardDescription, CardContent, CardFooter } from './components/ui'
import { VideoPlayer } from './components/VideoPlayer'
import { AudioWaveform } from './components/AudioWaveform'
import { FeatureImportanceChart } from './components/FeatureImportanceChart'

interface BoundingBox {
  x: number
  y: number
  width: number
  height: number
  label: string
}

export default function App() {
  const [file, setFile] = useState<File | null>(null)
  const [result, setResult] = useState<string>('')
  const [isInferring, setIsInferring] = useState(false)
  const [lastInferRaw, setLastInferRaw] = useState<any>(null)
  const [boundingBoxes, setBoundingBoxes] = useState<BoundingBox[]>([])
  const [annotationsData, setAnnotationsData] = useState<any>(null)
  const [mediaUrl, setMediaUrl] = useState<string>('')
  const [sharedTime, setSharedTime] = useState<number>(0)
  const [isTimeSyncEnabled, setIsTimeSyncEnabled] = useState(true)

  const handleTimeChange = (time: number) => {
    if (isTimeSyncEnabled) {
      setSharedTime(time)
    }
  }

  const handleFileChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    if (e.target.files) {
      const selectedFile = e.target.files[0]
      setFile(selectedFile)
      setMediaUrl(URL.createObjectURL(selectedFile))
      setResult('')
      setBoundingBoxes([])
      setSharedTime(0) // Reset shared time when new file is loaded
    }
  }

  const handleInfer = async () => {
    if (!file) return
    setIsInferring(true)
    setResult('Analyzing...')
    try {
      // Quick check to ensure backend is reachable
      try {
        const h = await fetch(((import.meta as any)?.env?.VITE_BACKEND_URL || 'http://localhost:8000') + '/health')
        console.log('Backend /health check status', h.status)
      } catch (hErr) {
        console.warn('Backend /health check failed', hErr)
      }
      const inferUrl = (((import.meta as any)?.env?.VITE_BACKEND_URL) || 'http://localhost:8000') + '/infer/?explain=true'
      const fd = new FormData()
      fd.append('file', file)
      const resp = await fetch(inferUrl, { method: 'POST', body: fd, mode: 'cors' })
      console.log('Infer HTTP response status:', resp.status, 'ok:', resp.ok)
      const ctype = resp.headers.get('content-type')
      console.log('Infer response content-type:', ctype)
      let data: any = null
      try {
        data = await resp.json()
      } catch (jsonErr) {
        console.warn('Failed to parse JSON from /infer response; trying to read text:', jsonErr)
        try {
          const text = await resp.text()
          console.warn('Raw /infer text response (first 1k chars):', text.slice(0, 1024))
        } catch (textErr) {
          console.error('Failed to read /infer response text:', textErr)
        }
        throw jsonErr
      }
      console.log('Infer response (parsed JSON):', data)
      setLastInferRaw(data)
      // Set results and map bounding boxes
      // Response format: { bounding_box: [{ frame, x1, y1, x2, y2, confidence }], inference: [{ class, frame }] }
      if (data?.inference && Array.isArray(data.inference) && data.inference.length > 0) {
        const top = data.inference[0]
        setResult(`Top: ${top.class} (frame ${top.frame})`)
      } else {
        setResult('No inference returned')
      }

      // Create class mapping for frames
      const classMap: Record<number, string> = {}
      if (data?.inference && Array.isArray(data.inference) && Array.isArray(data?.bounding_box)) {
        // Sort inference by frame
        const sortedInference = data.inference.sort((a: any, b: any) => (a.frame || 0) - (b.frame || 0))
        
        // Get all unique frame numbers from bounding boxes
        const allFrames = [...new Set(data.bounding_box.map(b => Number(b.frame || 0)))].sort((a, b) => a - b)
        const maxFrame = Math.max(...allFrames)
        
        for (let i = 0; i < sortedInference.length; i++) {
          const current = sortedInference[i]
          const next = sortedInference[i + 1]
          const startFrame = Number(current.frame || 0)
          const endFrame = next ? Number(next.frame || 0) - 1 : maxFrame // Last class applies to all remaining frames
          
          // Map the class to all frames from startFrame to endFrame
          for (let frame = startFrame; frame <= endFrame; frame++) {
            if (allFrames.includes(frame)) {
              classMap[frame] = current.class
            }
          }
        }
      }

      // Convert bounding_box list into UI boundingBoxes (use first entries as a global list)
      if (Array.isArray(data?.bounding_box)) {
        setBoundingBoxes([]) // Clear global boxes, use frame-keyed only
        
        // Set annotationsData for time-keyed rendering: group by frame
        const map: Record<number, any[]> = {}
        for (const b of data.bounding_box) {
          const frame = Number(b.frame || 0)
          const emotionClass = classMap[frame] || 'Unknown'
          const yoloConfidence = Math.round((b.confidence || 0) * 100)
          const box = { 
            x: b.x1, 
            y: b.y1, 
            width: (b.x2 - b.x1), 
            height: (b.y2 - b.y1), 
            label: `${emotionClass}\nconf ${yoloConfidence}%` 
          }
          if (!map[frame]) map[frame] = []
          map[frame].push(box)
        }
        console.log('Annotation map created with frames:', Object.keys(map).map(k => Number(k)).sort((a, b) => a - b))
        console.log('Class mapping:', classMap)
        setAnnotationsData(map)
        setLastInferRaw({ bounding_box: data.bounding_box, inference: data.inference, classMap })
      }
    } catch (err) {
      console.error('Infer API call failed', err)
      setResult('Inference failed')
    } finally {
      setIsInferring(false)
    }
  }

  const isVideo = file?.type.startsWith('video/')
  const isAudio = file?.type.startsWith('audio/')

  return (
    <div className="min-h-screen bg-gradient-to-br from-slate-50 to-slate-100">
      <div className="container mx-auto px-4 py-8 max-w-7xl">
        {/* Header */}
        <div className="text-center mb-12">
          <h1 className="text-4xl font-bold text-slate-900 mb-3">Multi-Modal Emotion Recognition</h1>
          <p className="text-lg text-slate-600 max-w-2xl mx-auto">
            Upload video or audio files to detect emotions using advanced AI analysis
          </p>
        </div>

        <div className="grid gap-8 lg:grid-cols-12">
          {/* Upload Section */}
          <div className="lg:col-span-4">
            <Card className="shadow-lg border-0 bg-white/80 backdrop-blur-sm">
              <CardHeader className="pb-4">
                <CardTitle className="flex items-center gap-2 text-xl">
                  <Icon name="Upload" className="h-5 w-5 text-blue-600" />
                  Upload Media
                </CardTitle>
                <CardDescription className="text-slate-600">
                  Select a video or audio file for emotion analysis
                </CardDescription>
              </CardHeader>
              <CardContent className="space-y-6">
                <div className="space-y-2">
                  <Label htmlFor="file-upload" className="text-sm font-medium text-slate-700">
                    Choose file
                  </Label>
                  <Input
                    id="file-upload"
                    type="file"
                    accept="video/mp4,video/webm,video/ogg,video/x-msvideo,video/quicktime,video/x-ms-wmv,video/x-flv,video/x-matroska,.mkv,.avi,.mov,.wmv,.flv,.webm,.m4v,.mpg,.mpeg,audio/*"
                    onChange={handleFileChange}
                    className="cursor-pointer"
                  />
                  {file && (
                    <p className="text-sm text-slate-500 flex items-center gap-1">
                      <Icon name="File" className="h-4 w-4" />
                      {file.name}
                    </p>
                  )}
                </div>
              </CardContent>
              <CardFooter className="pt-4">
                <Button
                  onClick={handleInfer}
                  disabled={!file || isInferring}
                  className="w-full h-11 text-base font-medium"
                  size="lg"
                >
                  <Icon name="Play" className="mr-2 h-4 w-4" />
                  {isInferring ? 'Analyzing...' : 'Analyze Emotion'}
                </Button>
              </CardFooter>
            </Card>
          </div>

          {/* Media Preview & Results */}
          <div className="lg:col-span-8 space-y-6">
            {/* Media Preview */}
            {mediaUrl && (
              <Card className="shadow-lg border-0 bg-white/80 backdrop-blur-sm">
                <CardHeader className="pb-4">
                  <CardTitle className="flex items-center gap-2 text-xl">
                    <Icon name="Video" className="h-5 w-5 text-blue-600" />
                    Media Preview
                  </CardTitle>
                </CardHeader>
                <CardContent className="p-0">
                  <div className="aspect-video bg-slate-900 rounded-lg overflow-hidden">
                    {isVideo && <VideoPlayer videoUrl={mediaUrl} boundingBoxes={boundingBoxes} annotations={annotationsData} externalTime={sharedTime} onTimeChange={handleTimeChange} />}
                    {isAudio && <div className="flex items-center justify-center h-full text-white">Audio file loaded</div>}
                  </div>
                </CardContent>
              </Card>
            )}

            {/* Audio Waveform */}
            {mediaUrl && (
              <Card className="shadow-lg border-0 bg-white/80 backdrop-blur-sm">
                <CardHeader className="pb-4">
                  <CardTitle className="flex items-center gap-2 text-xl">
                    <Icon name="Waveform" className="h-5 w-5 text-blue-600" />
                    Audio Waveform
                  </CardTitle>
                </CardHeader>
                <CardContent>
                  <AudioWaveform audioUrl={mediaUrl} externalTime={sharedTime} onTimeChange={handleTimeChange} />
                </CardContent>
              </Card>
            )}

            {/* Feature Importance Chart */}
            {lastInferRaw?.inference && lastInferRaw.inference.some((inf: any) => inf.feature_importance) && (
              <FeatureImportanceChart
                inferenceData={lastInferRaw.inference}
                currentTime={sharedTime}
              />
            )}

            {/* Results */}
            <Card className="shadow-lg border-0 bg-white/80 backdrop-blur-sm">
              <CardHeader className="pb-4">
                <CardTitle className="flex items-center gap-2 text-xl">
                  <Icon name="BarChart3" className="h-5 w-5 text-blue-600" />
                  Analysis Results
                </CardTitle>
                <CardDescription className="text-slate-600">
                  Emotion detection results will appear here
                </CardDescription>
              </CardHeader>
              <CardContent>
                {result ? (
                  <div className="space-y-4">
                    {/* Main Result */}
                    <div className="p-4 bg-green-50 border border-green-200 rounded-lg">
                      <div className="flex items-center gap-2">
                        <Icon name="CheckCircle" className="h-5 w-5 text-green-600" />
                        <p className="text-green-800 font-medium">{result}</p>
                      </div>
                    </div>

                    {/* Bounding Boxes Info */}
                    {annotationsData && Object.keys(annotationsData).length > 0 && (
                      <div className="p-4 bg-blue-50 border border-blue-200 rounded-lg">
                        <div className="flex items-center gap-2 mb-2">
                          <Icon name="Target" className="h-4 w-4 text-blue-600" />
                          <span className="text-sm font-medium text-blue-800">
                            Face Detection: {Object.keys(annotationsData).length} frames analyzed
                          </span>
                        </div>
                        <p className="text-xs text-blue-600">
                          Bounding boxes are displayed in real-time as the video plays
                        </p>
                      </div>
                    )}

                    {/* Feature Importance */}
                    {lastInferRaw?.inference && lastInferRaw.inference.some((inf: any) => inf.feature_importance) && (
                      <div className="p-4 bg-purple-50 border border-purple-200 rounded-lg">
                        <div className="flex items-center gap-2 mb-3">
                          <Icon name="BarChart3" className="h-4 w-4 text-purple-600" />
                          <span className="text-sm font-medium text-purple-800">
                            Feature Importance Analysis
                          </span>
                        </div>
                        <div className="grid gap-4 md:grid-cols-2">
                          {lastInferRaw.inference
                            .filter((inf: any) => inf.feature_importance)
                            .map((inf: any, idx: number) => (
                            <div key={idx} className="space-y-2">
                              <h4 className="text-xs font-medium text-purple-700">
                                Frame {inf.frame} - {inf.class}
                              </h4>
                              
                              {/* Video Features */}
                              <div>
                                <p className="text-xs text-purple-600 mb-1">Top Video Features:</p>
                                <div className="space-y-1">
                                  {inf.feature_importance.video.slice(0, 5).map((feat: any, featIdx: number) => (
                                    <div key={featIdx} className="flex justify-between text-xs">
                                      <span>Dim {feat.dimension}</span>
                                      <span className="font-mono">{feat.importance.toFixed(4)}</span>
                                    </div>
                                  ))}
                                </div>
                              </div>
                              
                              {/* Audio Features */}
                              <div>
                                <p className="text-xs text-purple-600 mb-1">Top Audio Features:</p>
                                <div className="space-y-1">
                                  {inf.feature_importance.audio.slice(0, 5).map((feat: any, featIdx: number) => (
                                    <div key={featIdx} className="flex justify-between text-xs">
                                      <span>Dim {feat.dimension}</span>
                                      <span className="font-mono">{feat.importance.toFixed(4)}</span>
                                    </div>
                                  ))}
                                </div>
                              </div>
                            </div>
                          ))}
                        </div>
                      </div>
                    )}
                  </div>
                ) : (
                  <div className="p-8 text-center text-slate-500">
                    <Icon name="Upload" className="h-12 w-12 mx-auto mb-4 opacity-50" />
                    <p className="text-lg font-medium mb-2">No results yet</p>
                    <p className="text-sm">Upload a file and click "Analyze Emotion" to get started</p>
                  </div>
                )}
              </CardContent>
            </Card>
          </div>
        </div>

        {/* Footer */}
        <footer className="text-center mt-16 text-slate-500">
          <p className="text-sm">Built with React, TypeScript, and shadcn/ui</p>
        </footer>
      </div>
    </div>
  )
}
