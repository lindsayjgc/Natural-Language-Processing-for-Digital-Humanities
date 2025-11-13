import { ProtectedRoute } from "@/components/auth/ProtectedRoute";
import { DocumentsView } from "@/components/documents/DocumentsView";

export const metadata = { title: "Documents" };

export default function DocumentsPage() {
  return (
    <ProtectedRoute>
      <DocumentsView />
    </ProtectedRoute>
  );
}
