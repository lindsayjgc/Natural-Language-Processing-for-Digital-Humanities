import { Badge } from "@/components/ui/badge";
import { Table, TableBody, TableCell, TableHead, TableHeader, TableRow } from "@/components/ui/table";

type DocumentRow = {
    name: string;
    genre: string;
    uploaded: string;
    status: "Analyzed" | "Analyzing";
};

const demoRows: DocumentRow[] = [
    { name: "The Great Gatsby", genre: "Fiction", uploaded: "2024-01-15", status: "Analyzed" },
    { name: "Sapiens: A Brief History of Humankind", genre: "Non-Fiction", uploaded: "2024-01-10", status: "Analyzing" },
    { name: "Hamlet", genre: "Drama", uploaded: "2023-12-20", status: "Analyzed" },
    { name: "The Waste Land", genre: "Poetry", uploaded: "2023-12-15", status: "Analyzed" },
    { name: "Dune", genre: "Sci-Fi", uploaded: "2023-12-01", status: "Analyzed" },
];

export function DocumentsTable() {
    return (
        <div className="mt-6 rounded-xl border border-black/10 bg-white overflow-hidden shadow-sm">
            <Table>
                <TableHeader>
                    <TableRow className="hover:bg-transparent">
                        <TableHead className="text-gray-600">Document Name</TableHead>
                        <TableHead className="text-gray-600">Genre</TableHead>
                        <TableHead className="text-gray-600">Uploaded</TableHead>
                        <TableHead className="text-gray-600">Status</TableHead>
                        <TableHead className="text-right text-gray-600">Actions</TableHead>
                    </TableRow>
                </TableHeader>
                <TableBody>
                    {demoRows.map((row) => (
                        <TableRow key={row.name} className="border-black/10">
                            <TableCell className="text-gray-900">{row.name}</TableCell>
                            <TableCell className="text-gray-700">
                                <Badge variant="secondary" className="bg-gray-100 text-gray-800 border-gray-200">
                                    {row.genre}
                                </Badge>
                            </TableCell>
                            <TableCell className="text-gray-700">{row.uploaded}</TableCell>
                            <TableCell className="text-gray-700">
                                {row.status === "Analyzed" ? (
                                    <span className="inline-flex items-center gap-2 text-emerald-600">
                                        <span className="size-2 rounded-full bg-emerald-500" /> Analyzed
                                    </span>
                                ) : (
                                    <span className="inline-flex items-center gap-2 text-amber-600">
                                        <span className="size-2 rounded-full bg-amber-500" /> Analyzing
                                    </span>
                                )}
                            </TableCell>
                            <TableCell className="text-right">
                                <a href="#" className="text-violet-600 hover:underline">
                                    View Analysis
                                </a>
                            </TableCell>
                        </TableRow>
                    ))}
                </TableBody>
            </Table>
        </div>
    );
}
