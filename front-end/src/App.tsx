import React, { useState } from 'react'
import { Button, Input, Label, Icon, Card, CardHeader, CardTitle, CardDescription, CardContent, CardFooter } from './components/ui'
import { VideoPlayer } from './components/VideoPlayer'
import { AudioWaveform } from './components/AudioWaveform'

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

  const handleFileChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    if (e.target.files) {
      const selectedFile = e.target.files[0]
      setFile(selectedFile)
      setMediaUrl(URL.createObjectURL(selectedFile))
      setResult('')
      setBoundingBoxes([])
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
      const inferUrl = (((import.meta as any)?.env?.VITE_BACKEND_URL) || 'http://localhost:8000') + '/infer/'
      const fd = new FormData()
      fd.append('file', file)
      // optional params can be added; example: window_size=5, explain=true
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

      // Convert bounding_box list into UI boundingBoxes (use first entries as a global list)
      if (Array.isArray(data?.bounding_box)) {
        // Don't populate global boundingBoxes - we'll use frame-keyed annotations instead
        // const boxes = data.bounding_box.map((b: any, idx: number) => {
        //   const w = (typeof b.x2 === 'number' && typeof b.x1 === 'number') ? (b.x2 - b.x1) : (b.width ?? 0)
        //   const h = (typeof b.y2 === 'number' && typeof b.y1 === 'number') ? (b.y2 - b.y1) : (b.height ?? 0)
        //   const label = b.label ?? `frame-${b.frame ?? idx}`
        //   return { x: b.x1, y: b.y1, width: w, height: h, label }
        // })
        // setBoundingBoxes(boxes)
        setBoundingBoxes([]) // Clear global boxes, use frame-keyed only
        
        // Set annotationsData for time-keyed rendering: group by frame
        const map: Record<number, any[]> = {}
        for (const b of data.bounding_box) {
          const frame = Number(b.frame || 0)
          const box = { x: b.x1, y: b.y1, width: (b.x2 - b.x1), height: (b.y2 - b.y1), label: `conf ${Math.round((b.confidence||0)*100)}%` }
          if (!map[frame]) map[frame] = []
          map[frame].push(box)
        }
        console.log('Annotation map created with frames:', Object.keys(map).map(k => Number(k)).sort((a, b) => a - b))
        setAnnotationsData(map)
        setLastInferRaw({ bounding_box: data.bounding_box, inference: data.inference })
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
    <div className="min-h-screen bg-gradient-to-br from-blue-50 to-indigo-100 p-8">
      <div className="mx-auto max-w-6xl">
        <header className="text-center mb-8">
          <h1 className="text-4xl font-bold text-gray-900 mb-2">Multi-Modal Emotion Recognition</h1>
          <p className="text-lg text-gray-600">Upload video or audio to detect emotions using AI</p>
        </header>

        <div className="grid gap-6 lg:grid-cols-3">
          <div className="lg:col-span-1">
            <Card>
              <CardHeader>
                <CardTitle className="flex items-center gap-2">
                  <Icon name="Upload" className="h-5 w-5" />
                  Upload Media
                </CardTitle>
                <CardDescription>
                  Select a video or audio file for emotion analysis
                </CardDescription>
              </CardHeader>
              <CardContent>
                <div className="space-y-4">
                  <div>
                    <Label htmlFor="file-upload">Choose file</Label>
                    <Input
                      id="file-upload"
                      type="file"
                      accept="video/mp4,video/webm,video/ogg,video/x-msvideo,video/quicktime,video/x-ms-wmv,video/x-flv,video/x-matroska,.mkv,.avi,.mov,.wmv,.flv,.webm,.m4v,.mpg,.mpeg,audio/*"
                      onChange={handleFileChange}
                      className="mt-1"
                    />
                    {file && (
                      <p className="text-sm text-gray-500 mt-1">
                        Selected: {file.name}
                      </p>
                    )}
                  </div>

                  {/* Test buttons for debugging */}
                  <div className="border-t pt-4">
                    <Label className="text-sm font-medium text-gray-700 mb-2 block">Test Files (Debug)</Label>
                    <div className="space-y-2">
                      <Button
                        onClick={() => {
                          setMediaUrl('http://localhost:5173/test.mp4')
                          setFile(null)
                          setResult('')
                          setBoundingBoxes([])
                        }}
                        variant="outline"
                        size="sm"
                        className="w-full"
                      >
                        Test MP4 File
                      </Button>
                      <Button
                        onClick={() => {
                          setMediaUrl('http://localhost:5173/test.flv')
                          setFile(null)
                          setResult('')
                          setBoundingBoxes([])
                        }}
                        variant="outline"
                        size="sm"
                        className="w-full"
                      >
                        Test FLV File
                      </Button>
                    </div>
                  </div>
                </div>
              </CardContent>
              <CardFooter>
                <Button
                  onClick={handleInfer}
                  disabled={!file || isInferring}
                  className="w-full"
                >
                  <Icon name="Play" className="mr-2 h-4 w-4" />
                  {isInferring ? 'Analyzing...' : 'Analyze Emotion'}
                </Button>
              </CardFooter>
            </Card>
          </div>

          <div className="lg:col-span-2 space-y-6">
            {mediaUrl && (
              <Card>
                <CardHeader>
                  <CardTitle>Media Preview</CardTitle>
                </CardHeader>
                <CardContent>
                  {isVideo && <VideoPlayer videoUrl={mediaUrl} boundingBoxes={boundingBoxes} annotations={annotationsData} />}
                  {isAudio && <AudioWaveform audioUrl={mediaUrl} />}
                </CardContent>
                {lastInferRaw && (
                  <CardContent>
                    <div className="mt-4 p-2 bg-gray-50 border rounded text-xs overflow-auto max-h-40">
                      <strong>Last /infer response:</strong>
                      <pre className="whitespace-pre-wrap">{JSON.stringify(lastInferRaw, null, 2)}</pre>
                    </div>
                  </CardContent>
                )}
              </Card>
            )}

            <Card>
              <CardHeader>
                <CardTitle className="flex items-center gap-2">
                  <Icon name="BarChart3" className="h-5 w-5" />
                  Results
                </CardTitle>
                <CardDescription>
                  Emotion detection results will appear here
                </CardDescription>
              </CardHeader>
              <CardContent>
                {result ? (
                  <div className="p-4 bg-green-50 border border-green-200 rounded-md">
                    <p className="text-green-800 font-medium">{result}</p>
                    {boundingBoxes.length > 0 && (
                      <div className="mt-4">
                        <h4 className="font-semibold mb-2">Detected Faces:</h4>
                        <ul className="space-y-1">
                          {boundingBoxes.map((box, index) => (
                            <li key={index} className="text-sm">
                              {box.label} at ({box.x}, {box.y})
                            </li>
                          ))}
                        </ul>
                      </div>
                    )}
                    {annotationsData && (
                      <div className="mt-4">
                        <h4 className="font-semibold mb-2">Annotations (from JSON):</h4>
                        <ul className="space-y-1">
                          {((Array.isArray(annotationsData) ? annotationsData : annotationsData?.bounding_box) || []).map((a: any, idx: number) => (
                            <li key={idx} className="text-sm">
                              {a.label ?? `Box ${idx}`} at ({a.x ?? (a?.bbox?.[0] ?? '-')}, {a.y ?? (a?.bbox?.[1] ?? '-')})
                            </li>
                          ))}
                        </ul>
                      </div>
                    )}
                  </div>
                ) : (
                  <div className="p-4 bg-gray-50 border border-gray-200 rounded-md text-center text-gray-500">
                    No results yet. Upload a file and click Analyze.
                  </div>
                )}
              </CardContent>
            </Card>
          </div>
        </div>

        <footer className="text-center mt-8 text-gray-500">
          <p>Built with React, TypeScript, and shadcn/ui</p>
        </footer>
      </div>
    </div>
  )
}
