import React from 'react';
import './MessageBox.css'

import TextField from '@mui/material/TextField'
import Button from '@mui/material/Button'
import TelegramIcon from '@mui/icons-material/Telegram';
import Grid from '@mui/material/Grid';

const MessageBox =React.forwardRef((props,ref)=>{
    
    const {handleClick} = props
    
    return (
        <form noValidate onSubmit={handleClick}>
            <Grid container className='box'>
                <Grid item xs={10}>
                    <TextField
                        placeholder='Type a message...'
                        variant="standard"
                        fullWidth
                        inputRef={ref}
                        InputProps={{
                            disableUnderline: true,
                        }}
                        style={{margin:'0.5rem'}}
                    />
                </Grid>
                <Grid item xs={2}>
                    <Button type="submit">
                        <TelegramIcon color="disabled" fontSize="large" />
                    </Button>
                </Grid>
            </Grid>
        </form>
    );
})
export default MessageBox