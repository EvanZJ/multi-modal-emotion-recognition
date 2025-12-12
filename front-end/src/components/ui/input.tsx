import * as React from 'react'
import { Slot } from '@radix-ui/react-slot'
import { cn } from '../../lib/utils'

export interface InputProps extends React.InputHTMLAttributes<HTMLInputElement> {
  asChild?: boolean
}

const Input = React.forwardRef<HTMLInputElement, InputProps>(
  ({ className, asChild = false, ...props }, ref) => {
    const Comp: any = asChild ? Slot : 'input'
    return <Comp ref={ref} className={cn('flex h-10 w-full rounded-md border border-gray-200 bg-transparent px-3 py-2 text-sm placeholder:text-muted-foreground focus:outline-none focus:ring-2 focus:ring-blue-500', className)} {...props} />
  }
)
Input.displayName = 'Input'

export default Input
