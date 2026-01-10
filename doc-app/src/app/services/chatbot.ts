export async function sendToChatbot(prompt: string, fileUrl?: string) {
  const res = await fetch(process.env.NEXT_PUBLIC_CHATBOT_API_URL!, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ prompt, fileUrl }),
  });

  if (!res.ok) throw new Error("Chatbot failed");

  return res.json();
}
