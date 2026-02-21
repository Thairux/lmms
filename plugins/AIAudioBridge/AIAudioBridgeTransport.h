/*
 * AIAudioBridgeTransport.h - transport worker for AI audio bridge effect
 *
 * This file is part of LMMS - https://lmms.io
 *
 * This program is free software; you can redistribute it and/or
 * modify it under the terms of the GNU General Public
 * License as published by the Free Software Foundation; either
 * version 2 of the License, or (at your option) any later version.
 */

#ifndef LMMS_AI_AUDIO_BRIDGE_TRANSPORT_H
#define LMMS_AI_AUDIO_BRIDGE_TRANSPORT_H

#include <atomic>
#include <condition_variable>
#include <mutex>
#include <thread>
#include <vector>

#include <QString>
#include <QStringList>

#include "LmmsTypes.h"
#include "SampleFrame.h"

class QProcess;

namespace lmms
{

class AIAudioBridgeTransport
{
public:
	enum class TransportMode
	{
		Pipe = 0,
		SharedMemory = 1
	};

	enum class Backend
	{
		Local = 0,
		Gemini = 1,
		DeepSeek = 2
	};

	struct Config
	{
		TransportMode transportMode = TransportMode::Pipe;
		Backend backend = Backend::Local;
		QString command = "python";
		QString scriptPath = "scripts/ai_audio_bridge_server.py";
		QString apiKey;
		sample_rate_t sampleRate = 44100;
		ch_cnt_t channels = 2;
		int timeoutMs = 8;
		float drive = 1.0f;
	};

	AIAudioBridgeTransport();
	~AIAudioBridgeTransport();

	void updateConfig(const Config& cfg);
	void stop();

	void submit(const SampleFrame* input, fpp_t frames);
	bool consume(SampleFrame* output, fpp_t frames);

private:
	void workerLoop();
	bool processBlockPipe(const Config& cfg, QProcess& process,
		const std::vector<float>& in, std::vector<float>& out, fpp_t frames);
	static void processBlockSharedMemory(const Config& cfg, const std::vector<float>& in, std::vector<float>& out);
	static QString backendToString(Backend backend);

	std::atomic<bool> m_running = false;
	std::thread m_worker;

	std::mutex m_mutex;
	std::condition_variable m_cv;

	Config m_config;
	bool m_configDirty = false;

	std::vector<float> m_pendingInput;
	fpp_t m_pendingFrames = 0;
	bool m_hasPendingInput = false;

	std::vector<float> m_readyOutput;
	fpp_t m_readyFrames = 0;
	bool m_hasReadyOutput = false;
};

} // namespace lmms

#endif // LMMS_AI_AUDIO_BRIDGE_TRANSPORT_H
