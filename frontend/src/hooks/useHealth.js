import { useQuery } from "@tanstack/react-query";
import { getHealth } from "../lib/api";

/**
 * Polls the /api/v1/health endpoint every 30 seconds.
 * Use the returned `data`, `isLoading`, and `refetch` in the Header.
 */
export function useHealth() {
  return useQuery({
    queryKey: ["health"],
    queryFn: getHealth,
    refetchInterval: 30_000,
    retry: 1,
    staleTime: 15_000,
  });
}
