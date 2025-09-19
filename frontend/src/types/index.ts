export type SessionType = {
  session_id: string;
  title: string;
};

export type MessageType = {
  message_id: string;
  message: string;
  response: string;
  metadata: {
    follow_up_questions: string[];
  };
};

export type ChatHistoryType = {
  session_id: string;
  title: string;
  messages: MessageType[];
};
