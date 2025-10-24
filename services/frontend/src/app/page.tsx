import { useAuth } from "@/contexts/AuthContext";
import { DocumentsView } from "@/components/documents/DocumentsView";
import { ProtectedRoute } from "@/components/auth/ProtectedRoute";


export const metadata = { title: "LitLens" };

function LandingPage() {
  return (
    <div>
      <h1>This is where landing page would be </h1>
    </div>
  );
}

export default function Home() {
  return (
    <ProtectedRoute fallback={<LandingPage />}>
      <DocumentsView />
    </ProtectedRoute>
  );
}
