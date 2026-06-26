"use client";

import type { ToolUIPart } from "ai";
import type { ComponentProps, ReactNode } from "react";

import { Button } from "@/components/ui/button";
import { cn } from "@/lib/utils";
import { createContext, useContext } from "react";

/* ------------------------------------------------------------------ */
/* Context                                                             */
/* ------------------------------------------------------------------ */

type ToolUIPartApproval = ToolUIPart["approval"];

interface ConfirmationContextValue {
  approval: ToolUIPartApproval;
  state: ToolUIPart["state"];
}

const ConfirmationContext = createContext<ConfirmationContextValue | null>(null);

const useConfirmation = () => {
  const ctx = useContext(ConfirmationContext);
  if (!ctx) throw new Error("Confirmation components must be used within <Confirmation>");
  return ctx;
};

/* ------------------------------------------------------------------ */
/* <Confirmation>                                                      */
/* ------------------------------------------------------------------ */

export type ConfirmationProps = ComponentProps<"div"> & {
  approval?: ToolUIPartApproval;
  state: ToolUIPart["state"];
};

export const Confirmation = ({
  className,
  approval,
  state,
  children,
  ...props
}: ConfirmationProps) => {
  if (!approval || state === "input-streaming" || state === "input-available") {
    return null;
  }

  return (
    <ConfirmationContext.Provider value={{ approval, state }}>
      <div
        className={cn(
          "flex flex-col gap-2 rounded-md border p-3 text-sm",
          className,
        )}
        {...props}
      >
        {children}
      </div>
    </ConfirmationContext.Provider>
  );
};

/* ------------------------------------------------------------------ */
/* <ConfirmationTitle>                                                 */
/* ------------------------------------------------------------------ */

export type ConfirmationTitleProps = ComponentProps<"p">;

export const ConfirmationTitle = ({
  className,
  ...props
}: ConfirmationTitleProps) => (
  <p className={cn("inline-flex items-center gap-2", className)} {...props} />
);

/* ------------------------------------------------------------------ */
/* State-conditional wrappers                                          */
/* ------------------------------------------------------------------ */

export const ConfirmationRequest = ({ children }: { children?: ReactNode }) => {
  const { state } = useConfirmation();
  return state === "approval-requested" ? <>{children}</> : null;
};

export const ConfirmationAccepted = ({ children }: { children?: ReactNode }) => {
  const { approval, state } = useConfirmation();
  const show =
    approval?.approved === true &&
    (state === "approval-responded" ||
      state === "output-available" ||
      state === "output-denied");
  return show ? <>{children}</> : null;
};

export const ConfirmationRejected = ({ children }: { children?: ReactNode }) => {
  const { approval, state } = useConfirmation();
  const show =
    approval?.approved === false &&
    (state === "approval-responded" ||
      state === "output-available" ||
      state === "output-denied");
  return show ? <>{children}</> : null;
};

/* ------------------------------------------------------------------ */
/* Actions                                                             */
/* ------------------------------------------------------------------ */

export type ConfirmationActionsProps = ComponentProps<"div">;

export const ConfirmationActions = ({
  className,
  ...props
}: ConfirmationActionsProps) => {
  const { state } = useConfirmation();
  if (state !== "approval-requested") return null;

  return (
    <div
      className={cn("flex items-center justify-end gap-2 self-end", className)}
      {...props}
    />
  );
};

export type ConfirmationActionProps = ComponentProps<typeof Button>;

export const ConfirmationAction = (props: ConfirmationActionProps) => (
  <Button size="sm" type="button" {...props} />
);
