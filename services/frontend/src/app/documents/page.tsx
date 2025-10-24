import { DocumentsView } from "@/components/documents/DocumentsView";
import { ProtectedRoute } from "@/components/auth/ProtectedRoute";

export const metadata = { title: "Documents" };

export default function DocumentsPage() {
  return (
    <ProtectedRoute>
      <DocumentsView />
    </ProtectedRoute>
  );
}
