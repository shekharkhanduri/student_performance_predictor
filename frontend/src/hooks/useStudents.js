import { useQuery, useQueryClient } from "@tanstack/react-query";
import { getStudents } from "../lib/api";

/**
 * Fetches the paginated students list.
 * Results are cached and shared across all components that call this hook
 * with the same params — no duplicate network requests.
 */
export function useStudents(params = {}) {
  return useQuery({
    queryKey: ["students", params],
    queryFn: () => getStudents(params),
    staleTime: 10_000,
    retry: 1,
  });
}

/**
 * Returns an invalidation function that forces all useStudents queries
 * to refetch.  Call this after a successful upload or predict to refresh
 * the students list without passing callbacks through the component tree.
 */
export function useInvalidateStudents() {
  const queryClient = useQueryClient();
  return () => queryClient.invalidateQueries({ queryKey: ["students"] });
}
