module.exports = {
  apps: [
    {
      name: 'newwave-frontend',
      script: 'npm',
      args: 'start',
      cwd: '/app',
      instances: 1,
      exec_mode: 'fork',
      env: {
        NODE_ENV: process.env.ENV === 'dev' ? 'development' : 'production',
        ENV: process.env.ENV || 'prod',
        PORT: 3000
      },
      env_development: {
        NODE_ENV: 'development',
        ENV: 'dev',
        PORT: 3000
      },
      env_staging: {
        NODE_ENV: 'production',
        ENV: 'stg',
        PORT: 3000
      },
      env_production: {
        NODE_ENV: 'production',
        ENV: 'prod',
        PORT: 3000
      },
      watch: false,
      max_memory_restart: '1G',
      // error_file: './logs/err.log',
      // out_file: './logs/out.log',
      // log_file: './logs/combined.log',
      time: true,
      merge_logs: true,
      log_date_format: 'YYYY-MM-DD HH:mm Z'
    }
  ]
}; 