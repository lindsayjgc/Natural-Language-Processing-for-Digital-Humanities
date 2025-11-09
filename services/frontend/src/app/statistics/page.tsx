import { ProtectedRoute } from "@/components/auth/ProtectedRoute";
import { StatisticsView } from "@/components/statistics/StatisticsView";

export const metadata = { title: "Statistics" };

export default function StatisticsPage() {
  return (
    <ProtectedRoute>
      <StatisticsView />
    </ProtectedRoute>
  );
}
