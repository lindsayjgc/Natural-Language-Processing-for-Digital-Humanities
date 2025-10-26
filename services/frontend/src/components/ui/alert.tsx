import React from "react";

export const Alert = ({ children }: { children: React.ReactNode }) => {
  return <div className="alert alert-info">{children}</div>;
};

export const AlertDescription = ({
  children,
}: {
  children: React.ReactNode;
}) => {
  return <p>{children}</p>;
};
