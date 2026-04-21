import { useQuery, UseQueryOptions } from '@tanstack/react-query'

/**
 * Polling hook — wraps useQuery with a default refetch interval.
 * Stops polling when the window is hidden (saves API calls).
 */
export function usePolling<T>(
  key: string | string[],
  fn: () => Promise<T>,
  intervalMs = 5000,
  options?: Partial<UseQueryOptions<T>>
) {
  return useQuery<T>({
    queryKey: Array.isArray(key) ? key : [key],
    queryFn: fn,
    refetchInterval: intervalMs,
    refetchIntervalInBackground: false,
    ...options,
  })
}
