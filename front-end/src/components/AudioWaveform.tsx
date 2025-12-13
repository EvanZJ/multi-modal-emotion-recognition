import React, { useRef, useEffect, useState } from 'react'
import WaveSurfer from 'wavesurfer.js'
import { Button, Icon } from './ui'

interface AudioWaveformProps {
  audioUrl: string
  externalTime?: number
  onTimeChange?: (time: number) => void
}

export const AudioWaveform: React.FC<AudioWaveformProps> = ({ audioUrl, externalTime, onTimeChange }) => {
  const waveformRef = useRef<HTMLDivElement>(null)
  const wavesurferRef = useRef<WaveSurfer | null>(null)
  const [isPlaying, setIsPlaying] = useState(false)
  const [isReady, setIsReady] = useState(false)

  // Sync external time updates
  useEffect(() => {
    if (externalTime !== undefined && wavesurferRef.current && isReady) {
      const currentAudioTime = wavesurferRef.current.getCurrentTime()
      if (Math.abs(currentAudioTime - externalTime) > 0.1) {
        wavesurferRef.current.seekTo(externalTime / wavesurferRef.current.getDuration())
      }
    }
  }, [externalTime, isReady])

  useEffect(() => {
    if (waveformRef.current && audioUrl) {
      // Destroy previous instance
      if (wavesurferRef.current) {
        wavesurferRef.current.destroy()
      }

      wavesurferRef.current = WaveSurfer.create({
        container: waveformRef.current,
        waveColor: '#64748b',
        progressColor: '#0f172a',
        height: 80,
        normalize: true,
        backend: 'WebAudio',
        barWidth: 2,
        barRadius: 1,
      })

      wavesurferRef.current.load(audioUrl)

      wavesurferRef.current.on('ready', () => {
        setIsReady(true)
      })

      wavesurferRef.current.on('play', () => setIsPlaying(true))
      wavesurferRef.current.on('pause', () => setIsPlaying(false))
      wavesurferRef.current.on('finish', () => setIsPlaying(false))

      wavesurferRef.current.on('audioprocess', () => {
        if (wavesurferRef.current) {
          onTimeChange?.(wavesurferRef.current.getCurrentTime())
        }
      })

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
    <div className="space-y-3">
      <div className="bg-slate-50 border border-slate-200 rounded-lg p-4">
        <div ref={waveformRef} className="min-h-[100px] rounded-md" />
      </div>
      <div className="flex justify-center">
        <Button
          onClick={togglePlay}
          disabled={!isReady}
          variant="outline"
          size="sm"
          className="px-4 py-2"
        >
          <Icon name={isPlaying ? 'Pause' : 'Play'} className="mr-2 h-4 w-4" />
          {isPlaying ? 'Pause' : 'Play'} Audio
        </Button>
      </div>
    </div>
  )
}