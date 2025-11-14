import { useState } from "react";
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogFooter,
  DialogHeader,
  DialogTitle,
} from "@/components/ui/dialog";
import { Button } from "@/components/ui/button";
import type { Document } from "@/lib/api";

type DeleteDocumentDialogProps = {
  document: Document | null;
  open: boolean;
  onOpenChange: (open: boolean) => void;
  onDelete: (documentId: string) => Promise<void>;
};

export function DeleteDocumentDialog({
  document,
  open,
  onOpenChange,
  onDelete,
}: DeleteDocumentDialogProps) {
  const [loading, setLoading] = useState(false);

  const handleDelete = async () => {
    if (!document) return;
    setLoading(true);
    try {
      await onDelete(document._id);
      onOpenChange(false);
    } catch (error) {
      console.error("Failed to delete document:", error);
    } finally {
      setLoading(false);
    }
  };

  const handleOpenChange = (newOpen: boolean) => {
    if (!loading) {
      onOpenChange(newOpen);
    }
  };

  return (
    <Dialog open={open} onOpenChange={handleOpenChange}>
      <DialogContent className="w-[calc(100vw-2rem)] max-w-md sm:max-w-lg">
        <DialogHeader>
          <DialogTitle>Delete Document</DialogTitle>
          <DialogDescription className="space-y-2">
            <span className="block">Are you sure you want to delete</span>
            <span
              className="font-semibold block break-all text-foreground"
              title={document?.filename}
            >
              "{document?.filename}"
            </span>
            <span className="block">This action cannot be undone.</span>
          </DialogDescription>
        </DialogHeader>
        <DialogFooter className="flex flex-col-reverse sm:flex-row gap-2">
          <Button
            variant="outline"
            onClick={() => handleOpenChange(false)}
            disabled={loading}
            className="w-full sm:w-auto"
          >
            Cancel
          </Button>
          <Button
            variant="destructive"
            onClick={handleDelete}
            disabled={loading}
            className="bg-violet-600 hover:bg-violet-700 w-full sm:w-auto"
          >
            {loading ? "Deleting..." : "Delete"}
          </Button>
        </DialogFooter>
      </DialogContent>
    </Dialog>
  );
}