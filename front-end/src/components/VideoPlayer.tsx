import React, { useRef, useEffect, useState } from 'react'
import { FFmpeg } from '@ffmpeg/ffmpeg'
import { AudioWaveform } from './AudioWaveform'

interface BoundingBox {
  x: number
  y: number
  width: number
  height: number
  label: string
}

interface VideoPlayerProps {
  videoUrl: string
  boundingBoxes: BoundingBox[]
  fileName?: string
  annotations?: any
}

export const VideoPlayer: React.FC<VideoPlayerProps> = ({ videoUrl, boundingBoxes, fileName, annotations }) => {
  const [isFlvFile, setIsFlvFile] = useState(false)
  const videoRef = useRef<HTMLVideoElement>(null)
  const canvasRef = useRef<HTMLCanvasElement>(null)
  const [currentTime, setCurrentTime] = useState(0)
  const [duration, setDuration] = useState(0)
  const [isLoaded, setIsLoaded] = useState(false)
  const [error, setError] = useState<string>('')
  const [isConverting, setIsConverting] = useState(false)
  const [conversionProgress, setConversionProgress] = useState(0)
  const [convertedVideoUrl, setConvertedVideoUrl] = useState<string>('')
  const convertedUrlRef = useRef<string | null>(null)
  const [detectedCodec, setDetectedCodec] = useState<string | null>(null)
  const convertedSourceRef = useRef<string | null>(null)
  const [isStreamingFallback, setIsStreamingFallback] = useState(false)
  const ffmpegRef = useRef<FFmpeg | null>(null)
  const flvPlayerRef = useRef<any>(null)
  const [annotationBoxes, setAnnotationBoxes] = useState<BoundingBox[]>([])
  const [annotationMap, setAnnotationMap] = useState<Record<number, BoundingBox[]>>({})
  const [frameAnnotationBoxes, setFrameAnnotationBoxes] = useState<BoundingBox[]>([])
  const [showAnnotations, setShowAnnotations] = useState(true)
  const [fps, setFps] = useState<number>(30) // default to 30 FPS

  // Initialize FFmpeg
  useEffect(() => {
    const loadFFmpeg = async () => {
      try {
        const ffmpeg = new FFmpeg()
      
      } catch (error) {
        console.error('Failed to load FFmpeg:', error)
        setError('Failed to initialize video converter')
      }
    }

    loadFFmpeg()
  }, [])

  // Parse `annotations` prop into normalized BoundingBox[] format
  useEffect(() => {
    const parseArray = (arr: any[]): BoundingBox[] => {
      const out: BoundingBox[] = []
      arr.forEach((entry: any, idx: number) => {
        if (!entry) return
        if (entry.x !== undefined && entry.y !== undefined && (entry.width !== undefined || entry.w !== undefined)) {
          const width = (entry.width ?? entry.w ?? 0)
          const height = (entry.height ?? entry.h ?? 0)
          out.push({ x: Number(entry.x), y: Number(entry.y), width: Number(width), height: Number(height), label: String(entry.label ?? entry.name ?? `box-${idx}`) })
          return
        }
        if (Array.isArray(entry.bbox) && entry.bbox.length >= 4) {
          const [x, y, w, h] = entry.bbox
          out.push({ x: Number(x), y: Number(y), width: Number(w), height: Number(h), label: entry.label ?? entry.name ?? `box-${idx}` })
          return
        }
        if (Array.isArray(entry) && entry.length >= 4 && typeof entry[0] === 'number') {
          const [x, y, w, h] = entry
          out.push({ x: Number(x), y: Number(y), width: Number(w), height: Number(h), label: `box-${idx}` })
          return
        }
        if (entry.xmin !== undefined && entry.ymin !== undefined && entry.xmax !== undefined && entry.ymax !== undefined) {
          const x = Number(entry.xmin), y = Number(entry.ymin)
          const width = Number(entry.xmax) - x
          const height = Number(entry.ymax) - y
          out.push({ x, y, width, height, label: entry.label ?? entry.name ?? `box-${idx}` })
          return
        }
      })
      return out
    }

    try {
      if (!annotations) {
        setAnnotationBoxes([])
        return
      }
      let obj: any = annotations
      if (typeof annotations === 'string') {
        try { obj = JSON.parse(annotations) } catch (e) { obj = annotations }
      }
      let arrayCandidate: any[] | null = null
      if (Array.isArray(obj)) arrayCandidate = obj
      else if (Array.isArray(obj.bounding_box)) arrayCandidate = obj.bounding_box
      else if (Array.isArray(obj.bounding_boxes)) arrayCandidate = obj.bounding_boxes
      else if (Array.isArray(obj.boxes)) arrayCandidate = obj.boxes
      else if (Array.isArray(obj.annotations)) arrayCandidate = obj.annotations
      if (!arrayCandidate) {
        // If obj is an object with numeric keys mapping to arrays of boxes, treat it as time-keyed
        if (typeof obj === 'object' && obj !== null) {
          const numericMap: Record<number, BoundingBox[]> = {}
          let foundNumeric = false
          Object.keys(obj).forEach(k => {
            const v = obj[k]
            let maybeNum = Number(k)
            if (isNaN(maybeNum)) {
              // try to extract numeric timestamp from key names like 'frame_001' or 't_0.5'
              const m = k.match(/\d+(?:\.\d+)?/)
              if (m) maybeNum = Number(m[0])
            }
            if (!isNaN(maybeNum) && Array.isArray(v)) {
              const parsed = parseArray(v)
              if (parsed.length) {
                foundNumeric = true
                numericMap[maybeNum] = parsed
              }
            }
          })
          if (foundNumeric) {
            console.log('Parsed annotations as numeric-keyed map:', Object.keys(numericMap).length, 'frames')
            setAnnotationMap(numericMap)
            setAnnotationBoxes([])
            return
          }
        }
        setAnnotationBoxes([])
        return
      }
      const parsed = parseArray(arrayCandidate)
      setAnnotationMap({})
      setAnnotationBoxes(parsed)
    } catch (err) {
      console.warn('Failed to parse annotations prop', err)
      setAnnotationBoxes([])
    }
  }, [annotations])

  // Update frame annotations based on currentTime and annotationMap
  // The annotationMap keys are frame numbers, so we need to convert currentTime to frame number
  useEffect(() => {
    if (!annotationMap || Object.keys(annotationMap).length === 0) {
      setFrameAnnotationBoxes([])
      return
    }
    
    console.log('Frame update effect triggered. Current annotationMap keys:', Object.keys(annotationMap).length, 'currentTime:', currentTime.toFixed(2))
    
    // If fps is still default (30) and we have annotation data, try to calculate actual fps from it
    let effectiveFps = fps
    if (fps === 30 && duration > 0) {
      const maxFrame = Math.max(...Object.keys(annotationMap).map(k => Number(k)))
      if (maxFrame > 0) {
        // Estimate FPS: if we have frames up to frame N and duration D, FPS ≈ N / D
        const estimatedFps = maxFrame / duration
        console.log(`Estimated FPS from annotations: ${estimatedFps.toFixed(2)} (max frame ${maxFrame}, duration ${duration.toFixed(2)}s)`)
        effectiveFps = estimatedFps
      }
    }
    
    // Convert current time to frame number: frameNumber = currentTime * fps
    const currentFrame = Math.floor(currentTime * effectiveFps)
    
    // Check if we have annotations for this exact frame
    const keys = Object.keys(annotationMap).map(k => Number(k)).filter(k => !isNaN(k))
    
    // Also try nearby frames in case of rounding issues (±1 frame tolerance)
    let foundFrame = null
    for (let offset = 0; offset <= 1; offset++) {
      if (keys.includes(currentFrame + offset)) {
        foundFrame = currentFrame + offset
        break
      }
      if (offset > 0 && keys.includes(currentFrame - offset)) {
        foundFrame = currentFrame - offset
        break
      }
    }
    
    if (foundFrame !== null) {
      const boxes = annotationMap[foundFrame] || []
      if (foundFrame !== currentFrame) {
        console.log(`Frame ${currentFrame} (time ${currentTime.toFixed(2)}s, FPS ${effectiveFps.toFixed(2)}): Matched nearby frame ${foundFrame} with ${boxes.length} boxes`)
      } else {
        console.log(`Frame ${currentFrame} (time ${currentTime.toFixed(2)}s, FPS ${effectiveFps.toFixed(2)}): Exact match with ${boxes.length} boxes`)
      }
      setFrameAnnotationBoxes(boxes)
    } else {
      // Log every 30 frames to avoid spam
      if (currentFrame % 30 === 0) {
        console.log(`Frame ${currentFrame} (time ${currentTime.toFixed(2)}s, FPS ${effectiveFps.toFixed(2)}): No annotations. Available frames: [${keys.slice(0, 10).join(', ')}${keys.length > 10 ? '...' : ''}]`)
      }
      setFrameAnnotationBoxes([])
    }
  }, [currentTime, annotationMap, fps, duration])

  // Redraw canvas when frame annotations or bounding boxes change
  useEffect(() => {
    if (!videoRef.current || !canvasRef.current || !isLoaded) return

    const video = videoRef.current
    const canvas = canvasRef.current
    const ctx = canvas.getContext('2d')
    if (!ctx) return

    const rect = video.getBoundingClientRect()
    canvas.width = rect.width
    canvas.height = rect.height

    const scaleX = rect.width / video.videoWidth
    const scaleY = rect.height / video.videoHeight

    ctx.clearRect(0, 0, canvas.width, canvas.height)
    
    const drawBox = (box: BoundingBox, color: string) => {
      // Support both pixel coordinates or normalized (0..1) coordinates in annotations
      let scaledX: number, scaledY: number, scaledWidth: number, scaledHeight: number
      const maybeNormalized = box.x >= 0 && box.x <= 1 && box.width >= 0 && box.width <= 1
      if (maybeNormalized) {
        scaledX = box.x * rect.width
        scaledY = box.y * rect.height
        scaledWidth = box.width * rect.width
        scaledHeight = box.height * rect.height
      } else {
        scaledX = box.x * scaleX
        scaledY = box.y * scaleY
        scaledWidth = box.width * scaleX
        scaledHeight = box.height * scaleY
      }

      ctx.strokeStyle = color
      ctx.lineWidth = 1
      ctx.strokeRect(scaledX, scaledY, scaledWidth, scaledHeight)
      ctx.fillStyle = color
      ctx.font = '12px Arial'
      const label = box.label ?? 'box'
      ctx.fillRect(scaledX, scaledY - 20, ctx.measureText(label).width + 10, 20)
      ctx.fillStyle = '#ffffff'
      ctx.fillText(label, scaledX + 5, scaledY - 5)
    }
    
    if (Object.keys(annotationMap).length === 0) {
      ;(boundingBoxes || []).forEach(box => drawBox(box, '#3b82f6'))
    }
    // Only show per-frame annotations
    if (showAnnotations && frameAnnotationBoxes && frameAnnotationBoxes.length > 0) {
      console.log('Drawing', frameAnnotationBoxes.length, 'boxes for frame')
      ;(frameAnnotationBoxes || []).forEach(box => drawBox(box, '#10b981'))
    }
  }, [frameAnnotationBoxes, annotationMap, boundingBoxes, showAnnotations, isLoaded])

  // Convert FLV to MP4
  const convertFlvToMp4 = async (flvUrl: string): Promise<string> => {
    if (!ffmpegRef.current) {
      throw new Error('FFmpeg not initialized')
    }

    const ffmpeg = ffmpegRef.current
    setIsConverting(true)
    setConversionProgress(0)

    try {
      // if ffmpeg hasn't loaded successfully yet, try to load it (retry)
      if (!ffmpeg.loaded) {
        try {
          console.log('FFmpeg not loaded; attempting to (re)load core files')
          await ffmpeg.load({
            coreURL: '/ffmpeg/ffmpeg-core.js',
            wasmURL: '/ffmpeg/ffmpeg-core.wasm',
          })
          console.log('FFmpeg reloaded successfully inside convert')
        } catch (e) {
          console.error('FFmpeg reload attempt failed:', e)
        }
      }
      console.log('convertFlvToMp4: starting for', flvUrl)
      // Fetch the FLV file
      const response = await fetch(flvUrl)
      if (!response.ok) {
        throw new Error(`Failed to fetch FLV: ${response.status}`)
      }
      const flvBlob = await response.blob()
      const flvArrayBuffer = await flvBlob.arrayBuffer()

      // Write input file to FFmpeg virtual filesystem
      await ffmpeg.writeFile('input.flv', new Uint8Array(flvArrayBuffer))
      console.log('convertFlvToMp4: wrote input.flv to FS, byteLength:', flvArrayBuffer.byteLength)
      console.log('Wrote input.flv to ffmpeg FS, size:', flvArrayBuffer.byteLength)

      // Try to probe the file to detect codec
      let videoCodec: string | null = null
      try {
        const probeOut: any = await ffmpeg.ffprobe(['input.flv'])
        console.log('ffprobe raw output:', probeOut)
        if (typeof probeOut === 'string') {
          try {
            const parsed = JSON.parse(probeOut)
            const vstream = parsed?.streams?.find((s: any) => s.codec_type === 'video')
            videoCodec = vstream?.codec_name || null
          } catch (jsonErr) {
            // fallback parse: look for 'Video: xyz' string
            const m = /Video: ([A-Za-z0-9_]+)/.exec(probeOut)
            if (m) videoCodec = m[1]
          }
        } else if (typeof probeOut === 'object') {
          const vstream = probeOut?.streams?.find((s: any) => s.codec_type === 'video')
          videoCodec = vstream?.codec_name || null
        }
      } catch (probeErr) {
        console.warn('ffprobe not available or failed:', probeErr)
      }
      console.log('Detected video codec:', videoCodec)
      setDetectedCodec(videoCodec)

      // Try to copy streams to mp4 container first (fast)
      try {
        console.log('Attempting stream copy conversion (fast)')
        await ffmpeg.exec(['-i', 'input.flv', '-c', 'copy', 'output.mp4'])
      } catch (copyErr) {
        console.warn('Stream copy failed, falling back to re-encode', copyErr)
        // Try to re-encode to h264/aac (common). If not available, fall back to mpeg4
        try {
          await ffmpeg.exec([
            '-i', 'input.flv',
            '-c:v', 'libx264',
            '-c:a', 'aac',
            '-preset', 'fast',
            '-crf', '22',
            'output.mp4'
          ])
        } catch (reErr) {
          console.warn('Re-encode with libx264 failed, trying mpeg4 with audio copy', reErr)
          // Try mpeg4 with audio copy (avoid requiring aac if not available)
          await ffmpeg.exec([
            '-i', 'input.flv',
            '-c:v', 'mpeg4',
            '-c:a', 'copy',
            'output.mp4'
          ])
        }
      }

      // Read the converted file
      const outputData = await ffmpeg.readFile('output.mp4')
      console.log('convertFlvToMp4: readFile returned, type:', typeof outputData, 'length:', (outputData as any)?.length ?? (outputData as Uint8Array)?.byteLength)
      const outputByteLength = (outputData as any)?.length ?? (outputData as Uint8Array)?.byteLength ?? 0
      console.log('FFmpeg output byte length:', outputByteLength)
      if (!outputByteLength) {
        throw new Error('Converted file is empty')
      }
      const outputBlob = new Blob([outputData as any], { type: 'video/mp4' })
      console.log('Converted output size:', (outputData as any)?.length || (outputData as Uint8Array)?.byteLength || 0)
      const convertedUrl = URL.createObjectURL(outputBlob)
      console.log('convertFlvToMp4: created convertedUrl', convertedUrl)

      // Clean up virtual files
      try { await ffmpeg.deleteFile('input.flv') } catch (e) {}
      try { await ffmpeg.deleteFile('output.mp4') } catch (e) {}

      return convertedUrl
    } catch (error) {
      console.error('FLV to MP4 conversion failed:', error)
      throw new Error('Failed to convert FLV to MP4')
    } finally {
      setIsConverting(false)
      setConversionProgress(0)
    }
  }

  // Check if file is FLV and convert if needed (detect by extension, mime, or magic bytes)
  useEffect(() => {
    if (!videoUrl) return

    const run = async () => {
      let isFlv = false
      try {
        // Quick check by filename extension
        if (videoUrl.toLowerCase().includes('.flv')) {
          isFlv = true
        } else {
          // Try to get content-type from headers (HEAD request)
          try {
            const headResp = await fetch(videoUrl, { method: 'HEAD' })
            const ctype = headResp.headers.get('content-type')
            if (ctype && ctype.toLowerCase().includes('flv')) {
              isFlv = true
            }
          } catch (headErr) {
            // HEAD may be blocked by CORS, ignore and continue to fetch bytes
            console.warn('HEAD request failed for mime detection', headErr)
          }

          // If we still don't know, fetch the first bytes and check magic signature
          if (!isFlv) {
            try {
              // Try a range request first to avoid downloading the entire file
              const rangeResp = await fetch(videoUrl, { headers: { Range: 'bytes=0-9' } })
              if (rangeResp.ok || rangeResp.status === 206) {
                const rangeBuf = await rangeResp.arrayBuffer()
                const bytes = new Uint8Array(rangeBuf.slice(0, 3))
                const isMagic = bytes[0] === 0x46 && bytes[1] === 0x4c && bytes[2] === 0x56
                if (isMagic) {
                  isFlv = true
                }
              } else {
                // Fall back to full fetch if range isn't supported
                const r = await fetch(videoUrl)
                if (r.ok) {
                  const arrbuf = await r.arrayBuffer()
                  const bytes = new Uint8Array(arrbuf.slice(0, 3))
                  const isMagic = bytes[0] === 0x46 && bytes[1] === 0x4c && bytes[2] === 0x56
                  if (isMagic) isFlv = true
                }
              }
            } catch (err) {
              console.warn('Failed to fetch initial bytes for magic detection', err)
            }
          }
        }
      } catch (err) {
        console.warn('Error during FLV detection', err)
      }
      console.log('Video URL:', videoUrl, 'Is FLV:', isFlv)
      setIsFlvFile(isFlv)

      // Reset state for a new video
      setError('')
      setIsLoaded(false)
      setCurrentTime(0)
      setDuration(0)
      setConvertedVideoUrl('')
      setIsStreamingFallback(false)

      if (!isFlv) {
        // revoke any existing converted object URL
        if (convertedUrlRef.current) {
          try { URL.revokeObjectURL(convertedUrlRef.current) } catch (e) {}
          convertedUrlRef.current = null
        }
        setConvertedVideoUrl(videoUrl)
        return
      }

      // For FLV, ensure FFmpeg is loaded
      const waitForFFmpeg = async (timeout = 60000) => {
        const start = Date.now()
        while (!ffmpegRef.current || !ffmpegRef.current.loaded) {
          if (Date.now() - start > timeout) throw new Error('FFmpeg failed to initialize')
          await new Promise(r => setTimeout(r, 200))
        }
      }

      try {
        await waitForFFmpeg()
      } catch (e) {
        console.error('FFmpeg init timeout', e)
        setError('Video converter not ready')
        // Try FLV streaming fallback if possible
        try {
          const flvModule: any = await import('flv.js')
          const flvjs: any = flvModule?.default ?? flvModule
          if (flvjs && typeof flvjs.isSupported === 'function' && flvjs.isSupported()) {
            const player = flvjs.createPlayer({ type: 'flv', url: videoUrl })
            if (videoRef.current) {
              player.attachMediaElement(videoRef.current)
              player.load()
              const p = player.play && player.play()
              if (p && typeof p.then === 'function') p.catch((ev: any) => console.log('FLV streaming play blocked:', ev))
              flvPlayerRef.current = player
              setIsStreamingFallback(true)
              setError('')
            }
            return
          }
          
        } catch (err) {
          console.warn('flv.js fallback attempt failed during FFmpeg init timeout', err)
        }
          console.warn('flv.js does not expose the expected API on dynamic import')
        return
      }

      try {
        if (convertedSourceRef.current === videoUrl && convertedUrlRef.current) {
          console.log('Reusing previously converted URL')
          setConvertedVideoUrl(convertedUrlRef.current)
          setError('')
          return
        }

        const converted = await convertFlvToMp4(videoUrl)
        // revoke previous if present
        if (convertedUrlRef.current) {
          try { URL.revokeObjectURL(convertedUrlRef.current) } catch (e) {}
          convertedUrlRef.current = null
        }
        convertedUrlRef.current = converted
        convertedSourceRef.current = videoUrl
        setConvertedVideoUrl(converted)
        setError('')
      } catch (error) {
        console.error('Conversion error:', error)
        // As a fallback, try streaming with flv.js for quick playback (non-converted)
        setError('Conversion failed, attempting FLV streaming fallback')
        try {
          const mod: any = await import('flv.js')
          const flvjs: any = mod?.default ?? mod
          console.log('flv.js module loaded for fallback')
          if (flvjs && typeof flvjs.isSupported === 'function' && flvjs.isSupported()) {
            const player: any = flvjs.createPlayer({ type: 'flv', url: videoUrl })
            if (videoRef.current) {
              player.attachMediaElement(videoRef.current)
              player.load()
              const p = player.play && player.play()
              if (p && typeof p.then === 'function') p.catch((e: any) => console.log('FLV streaming play blocked:', e))
              flvPlayerRef.current = player
              setIsStreamingFallback(true)
              setError('')
              console.log('FLV streaming fallback started')
            }
            return
          }
        } catch (err) {
          console.warn('Flv.js fallback failed or not installed:', err)
        }
        setError('Failed to convert FLV to MP4')
      }
    }

    run()
  }, [videoUrl])

  // Force conversion button handler - convert any URL regardless of extension/mime
  const handleForceConvert = async () => {
    try {
      setError('')
      setIsConverting(true)
      const converted = await convertFlvToMp4(videoUrl)
      if (convertedUrlRef.current) {
        try { URL.revokeObjectURL(convertedUrlRef.current) } catch (e) {}
        convertedUrlRef.current = null
      }
      convertedUrlRef.current = converted
      setConvertedVideoUrl(converted)
      setIsStreamingFallback(false)
      setError('')
    } catch (e) {
      console.error('Force convert failed', e)
      setError('Force conversion failed')
    } finally {
      setIsConverting(false)
    }
  }

  // Handle native HTML5 video for non-FLV files and converted FLV
  useEffect(() => {
    if (!videoRef.current) return

    const srcToUse = isFlvFile ? convertedVideoUrl : videoUrl
    const isPlayerReady = !isFlvFile ? !!videoUrl : !!convertedVideoUrl || isStreamingFallback
    // If there's no source to use yet (e.g., FLV converting) and not streaming fallback, don't attach listeners
    if (!isPlayerReady) {
      // ensure the current video is unloaded
      try {
        if (videoRef.current?.src) {
          videoRef.current.removeAttribute('src')
          videoRef.current.load()
        }
      } catch (e) {
        // ignore
      }
      return
    }

    const video = videoRef.current
    // Make sure video ref uses the updated source
    if (!isStreamingFallback && video.src !== srcToUse) {
      console.log('video: setting src to', srcToUse)
      if (srcToUse) {
        video.src = srcToUse
        try {
          video.load()
          const vp = video.play && video.play()
          if (vp && typeof vp.then === 'function') vp.catch((err: any) => {})
        } catch (e) {
          // ignore
        }
      } else {
        video.removeAttribute('src')
        video.load()
      }
    }

    const canvas = canvasRef.current
    if (!canvas) return

    const ctx = canvas.getContext('2d')
    if (!ctx) return

    const handleLoadedMetadata = () => {
      setDuration(video.duration)
      setIsLoaded(true)
      setError('')
      
      // Try to detect FPS from video metadata
      // For browsers, we can estimate from webkitVideoDecodedByteCount/presentedFrameCount
      // But for simplicity, we'll calculate it from the video element properties if available
      try {
        // Method 1: Check if we have frame rate info from media tracks
        const videoTrack = (video as any).captureStream?.()?.getVideoTracks?.()[0]
        if (videoTrack && videoTrack.getSettings) {
          const settings = videoTrack.getSettings()
          if (settings.frameRate) {
            console.log('Detected FPS from video track:', settings.frameRate)
            setFps(settings.frameRate)
            return
          }
        }
      } catch (e) {
        // Silently ignore errors
      }
      
      // Method 2: If we previously detected codec info via ffprobe during conversion, we could use that
      // For now, use a smart default: most videos are 24, 25, 30, 60 fps
      // We can also try to estimate from converting the video
      console.log('FPS detection failed, using default 30 FPS')
      setFps(30)
    }

    const handleTimeUpdate = () => {
      setCurrentTime(video.currentTime)
    }

    const handleError = () => {
      // Ignore errors caused by raw FLV auto-loading before conversion
      if (isFlvFile && (isConverting || (!convertedVideoUrl && !isStreamingFallback))) return
      try {
        const mediaErr = video.error
        console.error('Video element error:', mediaErr)
      } catch (e) {}
      setError('Video format not supported')
      setIsLoaded(false)
    }

    const handleLoadedData = () => {
      setIsLoaded(true)
      setError('')
    }

    video.addEventListener('loadedmetadata', handleLoadedMetadata)
    video.addEventListener('loadeddata', handleLoadedData)
    video.addEventListener('timeupdate', handleTimeUpdate)
    video.addEventListener('error', handleError)

    return () => {
      video.removeEventListener('loadedmetadata', handleLoadedMetadata)
      video.removeEventListener('loadeddata', handleLoadedData)
      video.removeEventListener('timeupdate', handleTimeUpdate)
      video.removeEventListener('error', handleError)
      // Revoke converted object URL if we set one
      if (isFlvFile && convertedVideoUrl) {
        try {
          URL.revokeObjectURL(convertedVideoUrl)
        } catch (e) {
          // ignore
        }
      }
      // Destroy flv.js player if used
        if (flvPlayerRef.current) {
          try { flvPlayerRef.current.destroy() } catch (e) {}
          flvPlayerRef.current = null
          setIsStreamingFallback(false)
        }
    }
  }, [isFlvFile, convertedVideoUrl, videoUrl, boundingBoxes, isConverting, isStreamingFallback, frameAnnotationBoxes, annotationMap])

  // Revoke object URL on unmount
  useEffect(() => {
    return () => {
      if (convertedUrlRef.current) {
        try { URL.revokeObjectURL(convertedUrlRef.current) } catch (e) {}
        convertedUrlRef.current = null
      }
      if (convertedSourceRef.current) convertedSourceRef.current = null
    }
  }, [])

  const handleSeek = (e: React.ChangeEvent<HTMLInputElement>) => {
    const time = parseFloat(e.target.value)
    if (videoRef.current) {
      videoRef.current.currentTime = time
    }
  }

  return (
    <div className="space-y-4">
      <div className="relative bg-black rounded-md overflow-hidden">
        {isFlvFile && isConverting && (
          <div className="absolute inset-0 flex items-center justify-center bg-black bg-opacity-75 text-white z-10">
            <div className="text-center p-4">
              <div className="animate-spin rounded-full h-8 w-8 border-b-2 border-white mx-auto mb-4"></div>
              <p className="text-sm mb-2">Converting FLV to MP4...</p>
              {detectedCodec && (
                <p className="text-xs mb-2 opacity-75">Detected codec: {detectedCodec}</p>
              )}
              {detectedCodec && detectedCodec.toLowerCase().indexOf('vp6') !== -1 && (
                <p className="text-xs text-yellow-300 mb-2">Note: VP6 (Flash) codec detected; conversion may be slow or may fail in-browser. Consider server-side conversion for large files.</p>
              )}
              <div className="w-full bg-gray-700 rounded-full h-2">
                <div
                  className="bg-blue-600 h-2 rounded-full transition-all duration-300"
                  style={{ width: `${conversionProgress}%` }}
                ></div>
              </div>
              <p className="text-xs mt-2 opacity-75">{conversionProgress}%</p>
            </div>
          </div>
        )}
        {isFlvFile && !isConverting && ffmpegRef.current && !ffmpegRef.current.loaded && (
          <div className="absolute inset-0 flex items-center justify-center bg-black bg-opacity-75 text-white z-10">
            <div className="text-center p-4">
              <div className="text-sm mb-2">Video detected as FLV — initializing converter...</div>
              <div className="text-xs opacity-75">If this takes too long, try server-side conversion or click "Force Convert" once initialized.</div>
            </div>
          </div>
        )}
        <video
          ref={videoRef}
          src={!isFlvFile ? videoUrl : (convertedVideoUrl || undefined)}
          controls
          className="w-full h-auto"
          preload="metadata"
        />
        {(convertedVideoUrl || !isFlvFile || isStreamingFallback) && (
          <div className="mt-3">
            <AudioWaveform audioUrl={(isFlvFile ? (convertedVideoUrl || videoUrl) : videoUrl) || ''} />
          </div>
        )}
        {(convertedVideoUrl || !isFlvFile || isStreamingFallback) && (
          <canvas
            ref={canvasRef}
            className="absolute top-0 left-0 pointer-events-none"
          />
        )}
        {isStreamingFallback && (
          <div className="absolute top-2 right-2 bg-yellow-500 text-black px-2 py-1 rounded text-xs">
            Streaming FLV (fallback)
          </div>
        )}
        {!isLoaded && !error && !isConverting && (
          <div className="absolute inset-0 flex items-center justify-center bg-black bg-opacity-50 text-white">
            <div className="text-center">
              <div className="animate-spin rounded-full h-8 w-8 border-b-2 border-white mx-auto mb-2"></div>
              Loading video...
            </div>
          </div>
        )}
        {error && (
          <div className="absolute inset-0 flex items-center justify-center bg-black bg-opacity-75 text-white">
            <div className="text-center p-4">
              <div className="text-red-400 text-lg mb-2">⚠️</div>
              <p className="text-sm">{error}</p>
                <p className="text-xs mt-2 opacity-75">
                {isFlvFile ? (
                  detectedCodec && detectedCodec.toLowerCase().indexOf('vp6') !== -1 ?
                  'FLV conversion failed: file uses On2 VP6 (Flash). Client-side conversion to MP4 may not be possible. Consider server-side conversion.' :
                  'FLV conversion failed. Try refreshing or check file format.'
                ) : 'Supported formats: MP4, WebM, OGG'}
              </p>
            </div>
          </div>
        )}
      </div>
      {!isConverting && (
        <div className="flex items-center space-x-4">
          <input
            type="range"
            min="0"
            max={duration || 100}
            value={currentTime}
            onChange={handleSeek}
            className="flex-1"
            disabled={!isLoaded || !!error}
          />
          <span className="text-sm text-gray-600">
            {Math.floor(currentTime)}s / {Math.floor(duration)}s
          </span>
        </div>
      )}
      {convertedVideoUrl && (
        <div className="flex items-center gap-4 mt-2">
          <a
            href={convertedVideoUrl}
            download={(fileName || 'converted').replace(/\.flv$/i, '.mp4')}
            className="px-3 py-1 bg-blue-600 text-white rounded text-sm"
          >
            Download converted
          </a>
        </div>
      )}
      <div className="flex items-center gap-4 mt-2">
        <button
          onClick={handleForceConvert}
          className="px-3 py-1 bg-green-600 text-white rounded text-sm"
          disabled={isConverting}
        >
          Force Convert
        </button>
        <button
          onClick={() => setShowAnnotations(v => !v)}
          className="px-3 py-1 bg-gray-700 text-white rounded text-sm"
        >
          {showAnnotations ? 'Hide' : 'Show'} annotations ({frameAnnotationBoxes.length})
        </button>
      </div>
    </div>
  )
}