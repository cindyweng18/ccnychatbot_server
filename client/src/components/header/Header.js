import React from 'react';
import './Header.css'

const Header =(props)=>{
    const {openModal} = props
    return (
        <div className='chat-header' onClick={openModal}>
            <p>Welcome to CCNY Chatbot!</p>
        </div>
    );
}
export default Header