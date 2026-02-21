/*
 * AIStudioView.cpp - UI for the AIStudio tool plugin
 *
 * Copyright (c) 2026
 *
 * This file is part of LMMS - https://lmms.io
 *
 * This program is free software; you can redistribute it and/or
 * modify it under the terms of the GNU General Public
 * License as published by the Free Software Foundation; either
 * version 2 of the License, or (at your option) any later version.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
 * General Public License for more details.
 *
 * You should have received a copy of the GNU General Public
 * License along with this program (see COPYING); if not, write to the
 * Free Software Foundation, Inc., 51 Franklin Street, Fifth Floor,
 * Boston, MA 02110-1301 USA.
 *
 */

#include "AIStudioView.h"

#include <QComboBox>
#include <QHBoxLayout>
#include <QJsonArray>
#include <QJsonDocument>
#include <QJsonObject>
#include <QLabel>
#include <QLineEdit>
#include <QProcess>
#include <QPushButton>
#include <QStringList>
#include <QTextEdit>
#include <QVBoxLayout>

#include "AIStudio.h"

namespace lmms::gui
{

namespace
{

constexpr const char* kProviderLocal = "local";
constexpr const char* kProviderGemini = "gemini";
constexpr const char* kProviderDeepSeek = "deepseek";

} // namespace


AIStudioView::AIStudioView(AIStudio* plugin) :
	ToolPluginView(plugin),
	m_plugin(plugin)
{
	setMinimumSize(760, 520);

	auto* mainLayout = new QVBoxLayout(this);

	auto* providerLayout = new QHBoxLayout();
	providerLayout->addWidget(new QLabel(tr("Provider:"), this));

	m_provider = new QComboBox(this);
	m_provider->addItem(tr("Local (free, no key)"), kProviderLocal);
	m_provider->addItem(tr("Gemini API (free tier)"), kProviderGemini);
	m_provider->addItem(tr("DeepSeek API"), kProviderDeepSeek);
	providerLayout->addWidget(m_provider, 1);

	providerLayout->addWidget(new QLabel(tr("API key:"), this));
	m_apiKey = new QLineEdit(this);
	m_apiKey->setPlaceholderText(tr("Optional: overrides env key"));
	m_apiKey->setEchoMode(QLineEdit::Password);
	providerLayout->addWidget(m_apiKey, 2);

	mainLayout->addLayout(providerLayout);

	mainLayout->addWidget(new QLabel(tr("Prompt"), this));
	m_prompt = new QTextEdit(this);
	m_prompt->setPlaceholderText(
		tr("Example: Build a dark trap drop in F# minor at 146 BPM, then suggest a modern loud mix chain."));
	mainLayout->addWidget(m_prompt, 1);

	m_generateButton = new QPushButton(tr("Generate"), this);
	mainLayout->addWidget(m_generateButton);

	m_status = new QLabel(tr("Ready"), this);
	mainLayout->addWidget(m_status);

	mainLayout->addWidget(new QLabel(tr("Response"), this));
	m_response = new QTextEdit(this);
	m_response->setReadOnly(true);
	mainLayout->addWidget(m_response, 1);

	connect(m_generateButton, &QPushButton::clicked, this, &AIStudioView::generate);
}


QString AIStudioView::selectedProviderName() const
{
	return m_provider->currentData().toString();
}


QString AIStudioView::resolveApiKey() const
{
	if (!m_apiKey->text().trimmed().isEmpty())
	{
		return m_apiKey->text().trimmed();
	}

	const auto provider = selectedProviderName();
	if (provider == kProviderGemini)
	{
		return QString::fromLocal8Bit(qgetenv("GEMINI_API_KEY")).trimmed();
	}
	if (provider == kProviderDeepSeek)
	{
		return QString::fromLocal8Bit(qgetenv("DEEPSEEK_API_KEY")).trimmed();
	}
	return QString();
}


QByteArray AIStudioView::buildRequestBody(const QString& prompt) const
{
	const auto provider = selectedProviderName();

	if (provider == kProviderGemini)
	{
		QJsonObject part;
		part.insert("text", prompt);

		QJsonObject content;
		content.insert("parts", QJsonArray{part});

		QJsonObject root;
		root.insert("contents", QJsonArray{content});

		return QJsonDocument(root).toJson(QJsonDocument::Compact);
	}

	QJsonObject systemMessage;
	systemMessage.insert("role", "system");
	systemMessage.insert("content",
		"Return practical LMMS-focused arrangement and mixing guidance with concise, actionable steps.");

	QJsonObject userMessage;
	userMessage.insert("role", "user");
	userMessage.insert("content", prompt);

	QJsonObject root;
	root.insert("model", "deepseek-chat");
	root.insert("messages", QJsonArray{systemMessage, userMessage});
	root.insert("temperature", 0.5);

	return QJsonDocument(root).toJson(QJsonDocument::Compact);
}


QString AIStudioView::parseResponse(const QByteArray& responseBody) const
{
	QJsonParseError parseError;
	const auto json = QJsonDocument::fromJson(responseBody, &parseError);
	if (parseError.error != QJsonParseError::NoError || !json.isObject())
	{
		return QString::fromUtf8(responseBody);
	}

	const auto root = json.object();
	const auto provider = selectedProviderName();
	if (provider == kProviderGemini)
	{
		const auto candidates = root.value("candidates").toArray();
		if (!candidates.isEmpty())
		{
			const auto parts = candidates.first().toObject().value("content").toObject().value("parts").toArray();
			QStringList textParts;
			for (const auto& value : parts)
			{
				const auto text = value.toObject().value("text").toString();
				if (!text.isEmpty())
				{
					textParts << text;
				}
			}
			if (!textParts.isEmpty())
			{
				return textParts.join("\n");
			}
		}
	}
	else if (provider == kProviderDeepSeek)
	{
		const auto choices = root.value("choices").toArray();
		if (!choices.isEmpty())
		{
			const auto content = choices.first().toObject().value("message").toObject().value("content").toString();
			if (!content.isEmpty())
			{
				return content;
			}
		}
	}

	return QString::fromUtf8(responseBody);
}


void AIStudioView::setBusy(bool busy)
{
	m_generateButton->setEnabled(!busy);
	m_provider->setEnabled(!busy);
}


void AIStudioView::generate()
{
	const auto prompt = m_prompt->toPlainText().trimmed();
	if (prompt.isEmpty())
	{
		m_status->setText(tr("Enter a prompt first."));
		return;
	}

	const auto provider = selectedProviderName();
	if (provider == kProviderLocal)
	{
		m_response->setPlainText(AIStudio::localFallbackResponse(prompt));
		m_status->setText(tr("Generated with local fallback mode."));
		return;
	}

	const auto apiKey = resolveApiKey();
	if (apiKey.isEmpty())
	{
		if (provider == kProviderGemini)
		{
			m_status->setText(tr("Missing API key. Set GEMINI_API_KEY or enter one."));
		}
		else
		{
			m_status->setText(tr("Missing API key. Set DEEPSEEK_API_KEY or enter one."));
		}
		return;
	}

	if (m_process != nullptr)
	{
		m_process->deleteLater();
		m_process = nullptr;
	}

	const auto payload = buildRequestBody(prompt);

	QString url;
	QStringList args = {"-sS", "-X", "POST"};
	args << "-H" << "Content-Type: application/json";

	if (provider == kProviderGemini)
	{
		url = "https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-flash:generateContent?key=" + apiKey;
	}
	else
	{
		url = "https://api.deepseek.com/chat/completions";
		args << "-H" << ("Authorization: Bearer " + apiKey);
	}

	args << url;
	args << "-d" << QString::fromUtf8(payload);

	m_process = new QProcess(this);
	connect(m_process, qOverload<int, QProcess::ExitStatus>(&QProcess::finished),
		this, [this](int exitCode, QProcess::ExitStatus) { requestFinished(exitCode); });
	connect(m_process, &QProcess::errorOccurred, this, &AIStudioView::requestErrored);

	setBusy(true);
	m_status->setText(tr("Requesting %1...").arg(m_provider->currentText()));
	m_process->start("curl", args);
}


void AIStudioView::requestFinished(int exitCode)
{
	setBusy(false);

	if (!m_process)
	{
		m_status->setText(tr("Request process missing."));
		return;
	}

	const auto stdoutText = m_process->readAllStandardOutput();
	const auto stderrText = QString::fromUtf8(m_process->readAllStandardError()).trimmed();

	if (exitCode != 0)
	{
		m_status->setText(tr("Request failed."));
		m_response->setPlainText(stderrText.isEmpty() ? tr("curl exited with code %1").arg(exitCode) : stderrText);
		return;
	}

	m_response->setPlainText(parseResponse(stdoutText));
	m_status->setText(tr("Done."));
}


void AIStudioView::requestErrored(QProcess::ProcessError)
{
	setBusy(false);
	if (!m_process)
	{
		m_status->setText(tr("Request failed."));
		return;
	}

	m_status->setText(tr("Request failed to start. Ensure curl is available on PATH."));
}

} // namespace lmms::gui
