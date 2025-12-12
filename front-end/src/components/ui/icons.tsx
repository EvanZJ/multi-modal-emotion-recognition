import * as React from 'react'
import * as Icons from 'lucide-react'

export function Icon({ name, className, ...props }: { name: keyof typeof Icons; className?: string }) {
  const Comp = Icons[name]
  if (!Comp) return null
  return <Comp className={className} {...props} />
}

export default Icon
