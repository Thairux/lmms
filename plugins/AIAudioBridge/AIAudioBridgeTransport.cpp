/*
 * AIAudioBridgeTransport.cpp - transport worker for AI audio bridge effect
 *
 * This file is part of LMMS - https://lmms.io
 *
 * This program is free software; you can redistribute it and/or
 * modify it under the terms of the GNU General Public
 * License as published by the Free Software Foundation; either
 * version 2 of the License, or (at your option) any later version.
 */

#include "AIAudioBridgeTransport.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <cstring>

#include <QDir>
#include <QProcess>
#include <QProcessEnvironment>

namespace lmms
{

namespace
{

constexpr std::array<char, 4> kMagic = {'A', 'I', 'B', 'R'};

struct PacketHeader
{
	char magic[4];
	std::uint32_t frames = 0;
	std::uint32_t channels = 0;
	float drive = 1.0f;
};

bool writeAll(QProcess& process, const char* data, qint64 bytes, int timeoutMs)
{
	qint64 written = 0;
	while (written < bytes)
	{
		const auto chunk = process.write(data + written, bytes - written);
		if (chunk < 0)
		{
			return false;
		}
		if (!process.waitForBytesWritten(timeoutMs))
		{
			return false;
		}
		written += chunk;
	}
	return true;
}

bool readAll(QProcess& process, char* data, qint64 bytes, int timeoutMs)
{
	qint64 read = 0;
	while (read < bytes)
	{
		if (process.bytesAvailable() <= 0 && !process.waitForReadyRead(timeoutMs))
		{
			return false;
		}
		const auto chunk = process.read(data + read, bytes - read);
		if (chunk < 0)
		{
			return false;
		}
		read += chunk;
	}
	return true;
}

} // namespace


AIAudioBridgeTransport::AIAudioBridgeTransport()
{
	m_running = true;
	m_worker = std::thread(&AIAudioBridgeTransport::workerLoop, this);
}


AIAudioBridgeTransport::~AIAudioBridgeTransport()
{
	stop();
}


void AIAudioBridgeTransport::stop()
{
	if (!m_running.exchange(false))
	{
		return;
	}
	m_cv.notify_all();
	if (m_worker.joinable())
	{
		m_worker.join();
	}
}


void AIAudioBridgeTransport::updateConfig(const Config& cfg)
{
	std::lock_guard<std::mutex> guard(m_mutex);
	if (m_config.transportMode == cfg.transportMode &&
		m_config.backend == cfg.backend &&
		m_config.command == cfg.command &&
		m_config.scriptPath == cfg.scriptPath &&
		m_config.apiKey == cfg.apiKey &&
		m_config.sampleRate == cfg.sampleRate &&
		m_config.channels == cfg.channels &&
		m_config.timeoutMs == cfg.timeoutMs &&
		std::abs(m_config.drive - cfg.drive) < 0.0001f)
	{
		return;
	}

	m_config = cfg;
	m_configDirty = true;
	m_cv.notify_one();
}


void AIAudioBridgeTransport::submit(const SampleFrame* input, fpp_t frames)
{
	if (frames == 0)
	{
		return;
	}

	const auto channels = static_cast<std::size_t>(DEFAULT_CHANNELS);
	std::vector<float> interleaved(frames * channels);
	for (fpp_t i = 0; i < frames; ++i)
	{
		interleaved[i * channels] = input[i].left();
		interleaved[i * channels + 1] = input[i].right();
	}

	{
		std::lock_guard<std::mutex> guard(m_mutex);
		m_pendingInput = std::move(interleaved);
		m_pendingFrames = frames;
		m_hasPendingInput = true;
	}

	m_cv.notify_one();
}


bool AIAudioBridgeTransport::consume(SampleFrame* output, fpp_t frames)
{
	std::lock_guard<std::mutex> guard(m_mutex);
	if (!m_hasReadyOutput || m_readyFrames != frames)
	{
		return false;
	}

	const auto channels = static_cast<std::size_t>(DEFAULT_CHANNELS);
	for (fpp_t i = 0; i < frames; ++i)
	{
		output[i].setLeft(m_readyOutput[i * channels]);
		output[i].setRight(m_readyOutput[i * channels + 1]);
	}
	m_hasReadyOutput = false;
	return true;
}


QString AIAudioBridgeTransport::backendToString(Backend backend)
{
	switch (backend)
	{
	case Backend::Gemini:
		return "gemini";
	case Backend::DeepSeek:
		return "deepseek";
	case Backend::Local:
	default:
		return "local";
	}
}


void AIAudioBridgeTransport::workerLoop()
{
	QProcess process;
	bool processStarted = false;
	Config activeConfig;

	while (m_running)
	{
		std::vector<float> input;
		fpp_t frames = 0;
		Config cfg;

		{
			std::unique_lock<std::mutex> lock(m_mutex);
			m_cv.wait(lock, [this] { return !m_running || m_hasPendingInput || m_configDirty; });
			if (!m_running)
			{
				break;
			}

			if (m_configDirty)
			{
				cfg = m_config;
				activeConfig = cfg;
				m_configDirty = false;
				if (processStarted)
				{
					process.kill();
					process.waitForFinished(200);
					processStarted = false;
				}
			}
			else
			{
				cfg = activeConfig;
			}

			if (!m_hasPendingInput)
			{
				continue;
			}

			input = std::move(m_pendingInput);
			frames = m_pendingFrames;
			m_hasPendingInput = false;
		}

		std::vector<float> output;
		bool ok = false;

		if (cfg.transportMode == TransportMode::SharedMemory)
		{
			processBlockSharedMemory(cfg, input, output);
			ok = true;
		}
		else
		{
			if (!processStarted)
			{
				QString scriptPath = cfg.scriptPath;
				if (QDir::isRelativePath(scriptPath))
				{
					scriptPath = QDir::current().absoluteFilePath(scriptPath);
				}

				QStringList args;
				args << scriptPath;
				args << "--backend" << backendToString(cfg.backend);
				args << "--sample-rate" << QString::number(cfg.sampleRate);
				args << "--channels" << QString::number(cfg.channels);

				process.setProgram(cfg.command);
				process.setArguments(args);

				auto env = QProcessEnvironment::systemEnvironment();
				if (!cfg.apiKey.isEmpty())
				{
					if (cfg.backend == Backend::Gemini)
					{
						env.insert("GEMINI_API_KEY", cfg.apiKey);
					}
					else if (cfg.backend == Backend::DeepSeek)
					{
						env.insert("DEEPSEEK_API_KEY", cfg.apiKey);
					}
				}
				process.setProcessEnvironment(env);
				process.start();
				processStarted = process.waitForStarted(1200);
			}

			if (processStarted)
			{
				ok = processBlockPipe(cfg, process, input, output, frames);
			}
		}

		if (!ok)
		{
			output = input;
		}

		std::lock_guard<std::mutex> guard(m_mutex);
		m_readyOutput = std::move(output);
		m_readyFrames = frames;
		m_hasReadyOutput = true;
	}

	if (processStarted)
	{
		process.kill();
		process.waitForFinished(200);
	}
}


void AIAudioBridgeTransport::processBlockSharedMemory(
	const Config& cfg, const std::vector<float>& in, std::vector<float>& out)
{
	out.resize(in.size());
	const float drive = std::max(0.1f, cfg.drive);
	for (std::size_t i = 0; i < in.size(); ++i)
	{
		out[i] = std::tanh(in[i] * drive);
	}
}


bool AIAudioBridgeTransport::processBlockPipe(
	const Config& cfg, QProcess& process, const std::vector<float>& in,
	std::vector<float>& out, fpp_t frames)
{
	PacketHeader header;
	std::memcpy(header.magic, kMagic.data(), kMagic.size());
	header.frames = static_cast<std::uint32_t>(frames);
	header.channels = static_cast<std::uint32_t>(cfg.channels);
	header.drive = cfg.drive;

	if (!writeAll(process, reinterpret_cast<const char*>(&header), sizeof(PacketHeader), cfg.timeoutMs))
	{
		return false;
	}

	const auto payloadBytes = static_cast<qint64>(in.size() * sizeof(float));
	if (!writeAll(process, reinterpret_cast<const char*>(in.data()), payloadBytes, cfg.timeoutMs))
	{
		return false;
	}

	PacketHeader responseHeader;
	if (!readAll(process, reinterpret_cast<char*>(&responseHeader), sizeof(PacketHeader), cfg.timeoutMs))
	{
		return false;
	}

	if (!std::equal(std::begin(responseHeader.magic), std::end(responseHeader.magic), std::begin(kMagic)))
	{
		return false;
	}
	if (responseHeader.frames != header.frames || responseHeader.channels != header.channels)
	{
		return false;
	}

	out.resize(in.size());
	if (!readAll(process, reinterpret_cast<char*>(out.data()), payloadBytes, cfg.timeoutMs))
	{
		return false;
	}
	return true;
}

} // namespace lmms
