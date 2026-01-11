const BASE_URL = process.env.NEXT_PUBLIC_CHATBOT_API_URL!;

export async function sendToChatbot(
  question: string,
  conversationId: string,
  fileUrl?: string | null
) {
  const res = await fetch(`${BASE_URL}/query`, {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
    },
    body: JSON.stringify({
      question,
      conversation_id: conversationId,
      file_url: fileUrl ?? null,
    }),
  });

  if (!res.ok) {
    throw new Error("Chatbot API failed");
  }

  return res.json();
}
