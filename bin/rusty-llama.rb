#!/usr/bin/env ruby

# USAGE example
# ruby rusty-llama.rb --model models/llama-2-7b-chat.Q4_0.gguf chat

require 'optparse'
require 'open3'

ENV['DYLD_LIBRARY_PATH'] = File.join(Dir.pwd, "bin")
# ENV['DYLD_LIBRARY_PATH'] = Dir.pwd
options = {
  temperature: 0.5,
  top_k: 40,
  top_p: 0.9
}

modes = %w[chat prompt csv]

parser = OptionParser.new do |opts|
  opts.banner = "Usage: ruby rusty_llama.rb [options] <mode> [mode_args]"

  opts.on("--model MODEL", "Path to model file") { |v| options[:model] = v }
  opts.on("--temperature FLOAT", Float, "Temperature (default 0.5)") { |v| options[:temperature] = v }
  opts.on("--top_k INT", Integer, "Top-k (default 40)") { |v| options[:top_k] = v }
  opts.on("--top_p FLOAT", Float, "Top-p (default 0.9)") { |v| options[:top_p] = v }
  opts.on("-h", "--help", "Prints this help") do
    puts opts
    puts "\nModes:"
    puts "  chat"
    puts "  prompt <prompt_text>"
    puts "  csv <csv_file> <output_file> <prompt_text>"
    exit
  end
end

parser.order!

if ARGV.empty?
  puts parser
  exit 1
end

mode = ARGV.shift

unless modes.include?(mode)
  puts "Unknown mode: #{mode}"
  puts parser
  exit 1
end

unless options[:model]
  puts "Error: --model option is required"
  exit 1
end

cmd = ["./bin/rusty_llama"]
cmd << "--model=#{options[:model]}"
cmd << "--temperature=#{options[:temperature]}"
cmd << "--top-k=#{options[:top_k]}"
cmd << "--top-p=#{options[:top_p]}"
cmd << mode

# Add mode-specific args
case mode
when "prompt"
  prompt_text = ARGV.shift
  unless prompt_text
    puts "prompt mode requires <prompt_text>"
    exit 1
  end
  cmd << prompt_text
when "csv"
  csv_file = ARGV.shift
  output_file = ARGV.shift
  prompt_text = ARGV.shift
  unless csv_file && output_file && prompt_text
    puts "csv mode requires <csv_file> <output_file> <prompt_text>"
    exit 1
  end
  cmd += [csv_file, output_file, prompt_text]
when "chat"
  # no extra args
end

puts "Running command: #{cmd.join(' ')}"

if mode == "chat"
  Open3.popen3(ENV.to_hash, *cmd) do |stdin, stdout, stderr, wait_thr|
#     out_thread = Thread.new do
#       until (line = stdout.gets).nil?
#         print line  # print immediately without adding extra newline
#       end
#     end
    out_thread = Thread.new do
      stdout.sync = true
      loop do
        char = stdout.read(1)
        break if char.nil?
        if char == '>'
          print "\n> "
        else
          print char
        end
        $stdout.flush
      end
    end

    err_thread = Thread.new do
      until (line = stderr.gets).nil?
        STDERR.print line
      end
    end

    begin
      while user_input = STDIN.gets
#         break if user_input.nil? || user_input.strip == "exit"

        stdin.puts user_input
        stdin.flush
      end
    rescue Errno::EPIPE
      # Rust binary closed stdin, exit input loop
    rescue Interrupt
      puts "\nInterrupted by user"
    end

    stdin.close rescue nil
    out_thread.join
    err_thread.join

    exit_status = wait_thr.value
    puts "\nRust process exited with status: #{exit_status.exitstatus}"
  end
else
  # For prompt or csv mode, just run and print output
  stdout_str, stderr_str, status = Open3.capture3(ENV.to_hash, *cmd)

  if status.success?
    puts stdout_str
  else
    STDERR.puts "Error running Rust binary:"
    STDERR.puts stderr_str
    exit 1
  end
end
