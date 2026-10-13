import { useEffect, useRef } from "react";

export function useLogStream(lines: string[], autoScroll: boolean) {
  const ref = useRef<HTMLPreElement>(null);
  useEffect(() => { if (autoScroll && ref.current) ref.current.scrollTop = ref.current.scrollHeight; }, [autoScroll, lines]);
  return ref;
}
