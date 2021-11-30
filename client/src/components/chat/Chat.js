
import React from 'react';
import './Chat.css'

const Chat = React.forwardRef((props,ref)=>{
    const {convo} = props

    return (
        <div className='paper'>
            {convo.map(item =>
                <>
                { item.mes?
                    <div className='chat-mes'>
                        {item.mes}
                    </div>
                    :
                    <div className='chat-res'>
                        {item.res}
                    </div>
                }
                </>
            )}
            <div ref={ref}></div>
        </div>
    );
})
export default Chat