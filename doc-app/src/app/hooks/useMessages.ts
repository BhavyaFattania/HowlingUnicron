"use client";

import { useEffect, useState } from "react";
import { doc, updateDoc } from "firebase/firestore";

import {
  addDoc,
  collection,
  onSnapshot,
  orderBy,
  query,
  serverTimestamp,
  where,
} from "firebase/firestore";
import { db } from "../services/firebase";

export function useMessages(chatId?: string | null) {
  const [messages, setMessages] = useState<any[]>([]);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    if (!chatId) {
      setMessages([]);
      setLoading(false);
      return;
    }

    setLoading(true);

    const q = query(
      collection(db, "messages"),
      where("chatId", "==", chatId),
      orderBy("createdAt", "asc")
    );

    const unsubscribe = onSnapshot(q, (snap) => {
      setMessages(
        snap.docs.map((d) => ({
          id: d.id,
          ...d.data(),
        }))
      );
      setLoading(false);
    });

    return unsubscribe;
  }, [chatId]);

  const addMessage = async (data: any) => {
    if (!chatId) return null;

    const docRef = await addDoc(collection(db, "messages"), {
      ...data,
      chatId,
      createdAt: serverTimestamp(),
    });

    return docRef; 
  };


  const updateMessage = async (id: string, data: any) => {
    await updateDoc(doc(db, "messages", id), data);
  };

  return { messages, addMessage, updateMessage,loading };
}
