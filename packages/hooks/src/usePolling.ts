import { useQuery, UseQueryOptions } from '@tanstack/react-query'

/**
 * Polling hook — wraps TanStack Query with a refetch interval.
 * Pauses polling when the browser tab is hidden (saves API calls).
 *
 * @example
 * const { data, isLoading } = usePolling('status', getStatus, 5000)
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
