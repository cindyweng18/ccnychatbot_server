import React, { useState, useEffect, useRef } from "react";
import axios from "axios";

import Container from "@mui/material/Container";

import Header from "./components/header/Header";
import Chat from "./components/chat/Chat";
import MessageBox from "./components/messageBox/MessageBox";
import ChatBubbleOutlineIcon from "@mui/icons-material/ChatBubbleOutline";
import Feedback from "./components/feedback/Feedback";
import RespondAnimation from "./components/respondAnimation/RespondAnimation"

const App = () => {
  const [convo, setConvo] = useState([]);

  const [botMessage, setBotMessage] = useState([]);
  const [isOpen, setIsOpen] = useState(false);
  const [isFeedbackOpen,setIsFeedbackOpen] = useState(false)
  const [isLoading, setIsLoading] = useState(false)

  const ref = useRef();

  useEffect(async () => {
    await axios.get("/chat/api/botmessage-list/").then(
      (response) => response.data.map((m) => setBotMessage(m.value)),
      (err) => console.error(err)
    );
  }, []);

  // Fixes scroll issue at the bottom of conversation.
  const messagesEndRef = useRef(null);
  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: "smooth" });
  };
  useEffect(() => {
    scrollToBottom();
  });

  const handleClick = async (e) => {
    e.preventDefault();

    let ab = ""

    if (ref.current.value.length > 0) {
      
      let newConvo = [...convo];
      newConvo.push({ mes: ref.current.value});
      
      setIsLoading(true)
      await axios.post ('/chat/api/botmessage-list/', {value: ref.current.value})
      .then (response => console.log (response))
      
      await axios
        .get("/chat/api/botmessage-list/")
        .then((response) => {
          response.data.map((m) => ab = m.value)
        });
        // send the user message to the backend
      setIsLoading(false)
      newConvo.push({ res: ab });
      setConvo(newConvo);
      console.log(ab);
      // empty the text message field
      ref.current.value = "";
      setIsFeedbackOpen(true)
    }
  };

  const handleFeedback = () => {
    //Send last user message to database after negative feedback
    console.log(convo[convo.length - 2]["mes"]);
    axios
      .post("/chat/api/message-list/", {
        value: convo[convo.length - 2]["mes"],
      })
      .then(function (response) {
        console.log(response);
      })
      .catch(function (error) {
        console.log(error);
      });
  };

  const openModal = () => {
    setIsOpen(!isOpen);
  };

  return (
    <>
      {isOpen ? (
        <Container
          component="main"
          maxWidth="xs"
          style={{ bottom: 150, right: 150, position: "absolute" }}
        >
          <div
            style={{
              boxShadow: "0px 0px 50px #a0a0a0",
              borderRadius: "1rem 1rem 1rem 1rem",
            }}
          >
            <Header openModal={openModal} />
            <Chat convo={convo} ref={messagesEndRef} />
            {isLoading &&
              <RespondAnimation/>
            }
            {isFeedbackOpen &&
              <Feedback setIsFeedbackOpen={setIsFeedbackOpen} handleFeedback={handleFeedback} />
            }
            <MessageBox handleClick={handleClick} ref={ref} />
          </div>
        </Container>
      ) : (
        <div onClick={openModal}>
          <ChatBubbleOutlineIcon
            sx={{
              color: "white",
              backgroundImage: "linear-gradient(to right, #6e0fc8, #cd49e6)",
              fontSize: "5rem",
              borderRadius: "50%",
              padding: "2rem",
              bottom: 50,
              right: 50,
              position: "absolute",
              boxShadow: "0px 0px 50px #a0a0a0",
            }}
          />
        </div>
      )}
    </>
  );
};

export default App;
