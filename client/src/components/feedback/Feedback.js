import React from 'react';
import './Feedback.css'

import ThumbDownOffAltIcon from '@mui/icons-material/ThumbDownOffAlt';
import ThumbUpOffAltIcon from '@mui/icons-material/ThumbUpOffAlt';
import Button from '@mui/material/Button'


const Feedback = (props)=>{

    const {handleFeedback,setIsFeedbackOpen } = props

    const handleClick =(e)=>{
        //TODO: once clicked, check if its up or down. If its down, save the question to database. Close the feedback component as well.
        if(e.currentTarget.value=="down"){
            handleFeedback()
            //close the component. I will do that later
            setIsFeedbackOpen(false)
        }
        else {
            //close the component. I will do that later
            setIsFeedbackOpen(false)

        }
    }
    return(
        <div className='text'>
            <span>
                <p>
                    Was this answer helpful?
                </p>
                <div>
                    <Button onClick={handleClick} value="up">
                        <ThumbUpOffAltIcon className='green'/>
                    </Button>
                    <Button onClick={handleClick} value="down">
                        <ThumbDownOffAltIcon className='red'/>
                    </Button>
                </div>
            </span>
        </div>
    )
}
export default Feedback