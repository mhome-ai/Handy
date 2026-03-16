import React from "react";

const HandyTextLogo = ({
  width,
  height,
  className,
}: {
  width?: number;
  height?: number;
  className?: string;
}) => {
  return (
    <div
      className={className}
      style={{
        width: width ? `${width}px` : undefined,
        height: height ? `${height}px` : undefined,
        display: "flex",
        alignItems: "center",
        fontSize: width ? `${Math.max(24, width * 0.28)}px` : "42px",
        fontWeight: 800,
        letterSpacing: "-0.04em",
        lineHeight: 1,
      }}
    >
      Footy
    </div>
  );
};

export default HandyTextLogo;
