import envVariables from '@/configs/env';
import axios from 'axios';

const axiosInstance = axios.create({
  baseURL: envVariables.apiUrl,
  headers: {
    'Content-Type': 'application/json',
    // Authorization will be added dynamically if using token from localStorage or cookie
  },
  timeout: 50000,
});

axiosInstance.interceptors.response.use(
  function (response) {
    // Any status code that lie within the range of 2xx cause this function to trigger
    // Do something with response data
    return response;
  },
  function (error) {
    // Any status codes that falls outside the range of 2xx cause this function to trigger
    // Do something with response error
    return Promise.reject(error);
  }
);

export default axiosInstance;
