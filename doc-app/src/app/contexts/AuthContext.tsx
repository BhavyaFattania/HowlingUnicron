"use client";

import {
  onAuthStateChanged,
  signInWithEmailAndPassword,
  signOut,
  signInWithPopup,
  GoogleAuthProvider,
} from "firebase/auth";
import { createContext, useContext, useEffect, useState } from "react";
import { auth } from "@/app/services/firebase";

const AuthContext = createContext<any>(null);
const googleProvider = new GoogleAuthProvider();

export function AuthProvider({ children }: any) {
  const [user, setUser] = useState<any>(null);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    return onAuthStateChanged(auth, (u) => {
      setUser(u);
      setLoading(false);
    });
  }, []);

  return (
    <AuthContext.Provider
      value={{
        user,
        loading,
        signIn: (email: string, pass: string) =>
          signInWithEmailAndPassword(auth, email, pass),
        signInWithGoogle: () =>
          signInWithPopup(auth, googleProvider),
        logout: () => signOut(auth),
      }}
    >
      {children}
    </AuthContext.Provider>
  );
}

export const useAuth = () => useContext(AuthContext);
