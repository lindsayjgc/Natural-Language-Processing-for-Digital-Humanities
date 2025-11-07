import { useState } from "react";
import { RefreshCw, MoreHorizontal } from "lucide-react";
import { useRouter } from "next/navigation";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import {
  Table,
  TableBody,
  TableCell,
  TableHead,
  TableHeader,
  TableRow,
} from "@/components/ui/table";
import type { Document } from "@/lib/api";

import {
  DropdownMenu,
  DropdownMenuTrigger,
  DropdownMenuContent,
  DropdownMenuItem,
  DropdownMenuSeparator,
} from "@/components/ui/dropdown-menu";
import { EditDocumentDialog } from "./EditDialog";
import { DeleteDocumentDialog } from "./DeleteDialog";

type DocumentsTableProps = {
  documents: Document[];
  loading: boolean;
  onRefresh: () => void;
  onUpdateDocument: (documentId: string, newFilename: string) => Promise<void>;
  onDeleteDocument: (documentId: string) => Promise<void>;
};

export function DocumentsTable({
  documents,
  loading,
  onRefresh,
  onUpdateDocument,
  onDeleteDocument,
}: DocumentsTableProps) {
  const router = useRouter();
  const [editingDocument, setEditingDocument] = useState<Document | null>(null);
  const [deletingDocument, setDeletingDocument] = useState<Document | null>(null);

  const formatDate = (dateString: string) => {
    return new Date(dateString).toLocaleDateString();
  };

  const handleRowClick = (doc: Document) => {
    if (doc.status === "completed") {
      router.push(`/documents/${doc._id}`);
    }
  };

  const handleEdit = (doc: Document) => {
    setEditingDocument(doc);
  };

  const handleDelete = (doc: Document) => {
    setDeletingDocument(doc);
  };

  const getStatusBadge = (status: string) => {
    switch (status) {
      case "completed":
        return (
          <Badge
            variant="default"
            className="bg-emerald-100 text-emerald-700 border-emerald-200"
          >
            <span className="size-2 rounded-full bg-emerald-500 mr-2" />
            Analyzed
          </Badge>
        );
      case "processing":
        return (
          <Badge
            variant="secondary"
            className="bg-amber-100 text-amber-700 border-amber-200"
          >
            <span className="size-2 rounded-full bg-amber-500 mr-2" />
            Analyzing
          </Badge>
        );
      case "failed":
        return (
          <Badge variant="destructive">
            <span className="size-2 rounded-full bg-red-500 mr-2" />
            Failed
          </Badge>
        );
      default:
        return (
          <Badge variant="outline" className="text-gray-600 border-gray-300">
            <span className="size-2 rounded-full bg-gray-500 mr-2" />
            Unknown
          </Badge>
        );
    }
  };

  if (loading) {
    return (
      <div className="mt-6 rounded-xl border border-black/10 bg-white overflow-hidden shadow-sm p-8 text-center">
        <div className="text-gray-600">Loading documents...</div>
      </div>
    );
  }

  return (
    <>
      <div className="mt-6 rounded-xl border border-black/10 bg-white overflow-hidden shadow-sm">
        <div className="flex justify-between items-center p-4 border-b border-gray-200">
          <h3 className="text-lg font-medium text-gray-900">Your Documents</h3>
          <Button
            onClick={onRefresh}
            variant="outline"
            size="sm"
            className="flex items-center gap-2"
          >
            <RefreshCw className="h-4 w-4" />
            Refresh
          </Button>
        </div>
        <Table>
          <TableHeader>
            <TableRow className="hover:bg-transparent">
              <TableHead className="text-gray-600">Document Name</TableHead>
              <TableHead className="text-gray-600">Uploaded</TableHead>
              <TableHead className="text-gray-600">Status</TableHead>
              <TableHead className="text-right text-gray-600">Actions</TableHead>
            </TableRow>
          </TableHeader>
          <TableBody>
            {documents.length === 0 ? (
              <TableRow>
                <TableCell colSpan={4} className="text-center text-gray-500 py-8">
                  No documents uploaded yet. Upload some files to get started!
                </TableCell>
              </TableRow>
            ) : (
              documents
                .filter((doc) => doc.status === "completed")
                .map((doc) => (
                  <TableRow
                    key={doc._id}
                    onClick={() => handleRowClick(doc)}
                    className={`border-black/10 ${
                      doc.status === "completed"
                        ? "cursor-pointer hover:bg-gray-50 transition-colors"
                        : "cursor-default"
                    }`}
                  >
                    <TableCell className="text-gray-900">
                      {doc.filename}
                    </TableCell>
                    <TableCell className="text-gray-700">
                      {formatDate(doc.uploaded_at)}
                    </TableCell>
                    <TableCell className="text-gray-700">
                      {getStatusBadge(doc.status)}
                    </TableCell>
                    <TableCell className="text-right">
                      <DropdownMenu>
                        <DropdownMenuTrigger asChild>
                          <Button
                            variant="ghost"
                            size="icon"
                            className="h-8 w-8 text-gray-500 hover:text-gray-800 cursor-pointer"
                            onClick={(e) => e.stopPropagation()}
                          >
                            <MoreHorizontal className="h-4 w-4" />
                          </Button>
                        </DropdownMenuTrigger>
                        <DropdownMenuContent align="end">
                          <DropdownMenuItem
                            onClick={(e) => {
                              e.stopPropagation();
                              handleRowClick(doc);
                            }}
                          >
                            View Analysis
                          </DropdownMenuItem>
                          <DropdownMenuSeparator />
                          <DropdownMenuItem
                            onClick={(e) => {
                              e.stopPropagation();
                              handleEdit(doc);
                            }}
                          >
                            Edit Name
                          </DropdownMenuItem>
                          <DropdownMenuItem
                            onClick={(e) => {
                              e.stopPropagation();
                              handleDelete(doc);
                            }}
                          >
                            Delete
                          </DropdownMenuItem>
                        </DropdownMenuContent>
                      </DropdownMenu>
                    </TableCell>
                  </TableRow>
                ))
            )}
          </TableBody>
        </Table>
      </div>

      <EditDocumentDialog
        document={editingDocument}
        open={!!editingDocument}
        onOpenChange={(open) => !open && setEditingDocument(null)}
        onSave={onUpdateDocument}
      />

      <DeleteDocumentDialog
        document={deletingDocument}
        open={!!deletingDocument}
        onOpenChange={(open) => !open && setDeletingDocument(null)}
        onDelete={onDeleteDocument}
      />
    </>
  );
}
