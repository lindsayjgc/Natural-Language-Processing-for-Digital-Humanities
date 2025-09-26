interface SectionBreakProps {
    title: string
}

export function SectionBreak({ title }: SectionBreakProps) {
    return (
        <>

            <h1 className="text-2xl font-semibold mt-8 mb-2">{title}</h1>
            <hr className="h-px my-4 mb-8 bg-gray-200 border-0 dark:bg-gray-700" />
        </>
    )
}