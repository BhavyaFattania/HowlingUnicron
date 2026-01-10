"use client";

import { signInAnonymously } from "firebase/auth";
import { auth } from "@/app/services/firebase";

export default function TestAuthPage() {
  return (
    <div style={{ padding: 40 }}>
      <button
        onClick={async () => {
          try {
            const res = await signInAnonymously(auth);
            console.log("ANON SIGN-IN OK", res.user.uid);
            alert("Anonymous auth WORKS");
          } catch (e) {
            console.error("ANON SIGN-IN FAILED", e);
            alert("Anonymous auth FAILED");
          }
        }}
      >
        Test Firebase Anonymous Auth
      </button>
    </div>
  );
}
