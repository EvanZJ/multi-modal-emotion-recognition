import React, { useRef, useEffect, useState } from 'react'
import WaveSurfer from 'wavesurfer.js'
import { Button } from './ui'

interface AudioWaveformProps {
  audioUrl: string
}

export const AudioWaveform: React.FC<AudioWaveformProps> = ({ audioUrl }) => {
  const waveformRef = useRef<HTMLDivElement>(null)
  const wavesurferRef = useRef<WaveSurfer | null>(null)
  const [isPlaying, setIsPlaying] = useState(false)
  const [isReady, setIsReady] = useState(false)

  useEffect(() => {
    if (waveformRef.current && audioUrl) {
      // Destroy previous instance
      if (wavesurferRef.current) {
        wavesurferRef.current.destroy()
      }

      wavesurferRef.current = WaveSurfer.create({
        container: waveformRef.current,
        waveColor: '#4f46e5',
        progressColor: '#1e1b4b',
        height: 80,
        normalize: true,
        backend: 'WebAudio',
      })

      wavesurferRef.current.load(audioUrl)

      wavesurferRef.current.on('ready', () => {
        setIsReady(true)
      })

      wavesurferRef.current.on('play', () => setIsPlaying(true))
      wavesurferRef.current.on('pause', () => setIsPlaying(false))
      wavesurferRef.current.on('finish', () => setIsPlaying(false))

      wavesurferRef.current.on('error', (err) => {
        console.error('WaveSurfer error:', err)
        setIsReady(false)
      })

      return () => {
        if (wavesurferRef.current) {
          wavesurferRef.current.destroy()
          wavesurferRef.current = null
        }
      }
    }
  }, [audioUrl])

  const togglePlay = () => {
    if (wavesurferRef.current && isReady) {
      wavesurferRef.current.playPause()
    }
  }

  return (
    <div className="space-y-2">
      <div ref={waveformRef} className="bg-gray-100 rounded-md p-2 min-h-[100px]" />
      <Button
        onClick={togglePlay}
        disabled={!isReady}
        variant="default"
      >
        {isPlaying ? 'Pause' : 'Play'}
      </Button>
    </div>
  )
}