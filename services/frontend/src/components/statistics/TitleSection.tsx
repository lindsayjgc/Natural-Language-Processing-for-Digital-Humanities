import { Edit, Save } from "lucide-react";
import { Button } from "../ui/button";

export function TitleSection() {
    return (
        <div className="flex flex-col lg:flex-row lg:items-center lg:justify-between gap-4">
            <div className="flex-1 min-w-0">
                <h1 className="text-2xl sm:text-3xl font-semibold mb-1 sm:mb-2 text-gray-900 truncate">
                    Statistics Dashboard
                </h1>
                <h2 className="text-lg sm:text-xl text-gray-600">
                    A Tale of Two Cities.docx
                </h2>
            </div>

            <div className="flex flex-col sm:flex-row gap-3 lg:flex-shrink-0">
                <Button className="text-gray-700 bg-white border border-gray-300">
                    <Edit size={16} />
                    <span>Edit Text</span>
                </Button>

                <Button className="px-4 text-white bg-blue-600 border border-blue-600 ">
                    <Save size={16} />
                    <span>Save</span>
                </Button>
            </div>
        </div>
    )
}